"""Inspect and retime an existing camera shot without changing its source."""
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid


MAX_FRAMES = 481
AUDIO_MODES = ('retime', 'preserve', 'mute')
# Compiled plan lines, as stored in WanGP's MP4 metadata (comment lines are stripped before saving).
PLAN_LINE = re.compile(r'(?m)^Camera plan: one continuous take, (\d+) frames at ([0-9.]+) fps;')
# H3 dialogue markup: <d>[Language] line</d> and (S1) speaker tags.
DIALOGUE = re.compile(r'<d>|\(S\d+\)')
HOLD_LINE = re.compile(r'(?m)^\[(\d+(?:\.\d+)?)s\u2013(\d+(?:\.\d+)?)s\] Hold the camera completely stationary')
SAMPLE_MODES = ('nearest', 'blend')


def _run(command, **options):
    return subprocess.run(command, check=True, capture_output=True,
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), **options)


def _tool(name):
    executable = shutil.which(name)
    if not executable:
        raise ValueError(f'{name} is required for camera timing tools. Install it on PATH, then restart WanGP.')
    return executable


def _integer(value, label):
    if isinstance(value, bool):
        raise ValueError(f'{label} must be a whole frame number.')
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f'{label} must be a whole frame number.') from error
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f'{label} must be a whole frame number.')
    return int(number)


def probe_video(source):
    if not source:
        raise ValueError('Choose a generated video first.')
    path = Path(source).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError('Choose a video file.')
    result = _run([_tool('ffprobe'), '-v', 'error', '-count_frames', '-show_streams', '-show_frames',
                   '-show_entries', 'stream=index,codec_type,width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,sample_rate:frame=media_type,best_effort_timestamp_time',
                   '-of', 'json', str(path)], timeout=60)
    data = json.loads(result.stdout)
    video = next((stream for stream in data['streams'] if stream['codec_type'] == 'video'), None)
    if video is None:
        raise ValueError('The file has no video stream.')
    try:
        fps = Fraction(video['avg_frame_rate'])
        nominal = Fraction(video['r_frame_rate'])
        frames = int(video.get('nb_read_frames') or video['nb_frames'])
    except (ValueError, KeyError, ZeroDivisionError) as error:
        raise ValueError('The video must have a readable, constant frame rate.') from error
    if fps <= 0 or abs(float(fps-nominal)) > 1e-6:
        raise ValueError('Use a constant-frame-rate video for frame-exact timing.')
    if not 3 <= frames <= MAX_FRAMES:
        raise ValueError(f'Timing correction supports a single shot of 3 to {MAX_FRAMES} frames.')
    timestamps = [float(frame['best_effort_timestamp_time']) for frame in data.get('frames',[])
                  if frame.get('media_type') == 'video' and 'best_effort_timestamp_time' in frame]
    tolerance = max(.001, .005/float(fps))
    if len(timestamps) != frames or any(not math.isfinite(value) or abs(value-timestamps[0]-index/float(fps))>tolerance
                                       for index,value in enumerate(timestamps)):
        raise ValueError('Use a constant-frame-rate video; its actual frame timestamps are irregular or missing.')
    if int(video['width']) % 2 or int(video['height']) % 2:
        raise ValueError('The video width and height must be even for this MP4 export.')
    audio = next((stream for stream in data['streams'] if stream['codec_type'] == 'audio'), None)
    return dict(path=str(path), frames=frames, fps=float(fps), fps_fraction=str(fps),
                width=int(video['width']), height=int(video['height']), duration=frames/float(fps),
                has_audio=audio is not None, audio_sample_rate=int(audio.get('sample_rate', 48000)) if audio else None)


def frame_map(frame_count, source_marks, target_marks):
    """Inverse piecewise-linear map, with exact rational positions at every mark."""
    frames = _integer(frame_count, 'Frame count')
    source = [_integer(value, 'Source mark') for value in source_marks]
    target = [_integer(value, 'Target mark') for value in target_marks]
    if len(source) != len(target) or len(source) < 2:
        raise ValueError('Provide matching source and target boundary lists.')
    if frames < 3 or any(marks[0] != 0 or marks[-1] != frames-1 for marks in (source, target)):
        raise ValueError('The first and last video frames must stay fixed.')
    if any(any(left >= right for left, right in zip(marks, marks[1:])) for marks in (source, target)):
        raise ValueError('Arrival and departure must be ordered strictly inside the clip.')
    result = []
    for frame in range(frames):
        part = min(bisect_right(target, frame)-1, len(target)-2)
        position = source[part] + Fraction((frame-target[part])*(source[part+1]-source[part]), target[part+1]-target[part])
        result.append(position)
    return result


def held_path_frames(path, keyframe, frame_count):
    from .camera_plan import validate_path, keyframe_frame
    poses = validate_path(path)
    number = _integer(keyframe, 'Hold starting keyframe')
    if not 1 <= number < len(poses):
        raise ValueError('Choose the starting keyframe of a hold.')
    left, right = poses[number-1:number+1]
    if any(left[field] != right[field] for field in ('azimuth', 'elevation', 'distance')):
        raise ValueError('The selected keyframe must be followed by the same camera pose.')
    first, last = keyframe_frame(left, frame_count), keyframe_frame(right, frame_count)
    if not 0 < first < last < frame_count-1:
        raise ValueError('Choose an interior hold spanning at least one displayed frame.')
    return first, last


def extract_frame(source, one_based_frame):
    import cv2
    from PIL import Image
    info = probe_video(source)
    frame = _integer(one_based_frame, 'Source frame')-1
    if not 0 <= frame < info['frames']:
        raise ValueError(f"Choose a source frame between 1 and {info['frames']}.")
    capture = cv2.VideoCapture(info['path'])
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, pixels = capture.read()
        if not ok:
            raise ValueError('Could not decode the selected frame.')
        return Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)), info, frame
    finally:
        capture.release()


def background_motion(source, info=None):
    """Measure visible background displacement; this is not camera-pose recovery."""
    import cv2
    import numpy as np
    info = info or probe_video(source)
    capture = cv2.VideoCapture(info['path'])
    scale = min(1.0, 640/info['width'])
    previous = None
    pairs = []
    try:
        for frame in range(info['frames']):
            ok, pixels = capture.read()
            if not ok:
                raise ValueError(f'Could not decode source frame {frame+1}.')
            gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
            if scale < 1:
                gray = cv2.resize(gray, (round(info['width']*scale),round(info['height']*scale)))
            if previous is not None:
                h, w = gray.shape
                mask = np.zeros_like(gray)
                margin = max(2, round(30*scale))
                mask[margin:h-margin,margin:round(w*.32)] = 255
                mask[margin:h-margin,round(w*.68):w-margin] = 255
                points = cv2.goodFeaturesToTrack(previous, maxCorners=180, qualityLevel=.02, minDistance=5, mask=mask)
                median, tracked_count = None, 0
                if points is not None and len(points) >= 12:
                    tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None)
                    reverse, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous, tracked, None)
                    valid = (status.ravel()==1) & (back_status.ravel()==1) & (np.linalg.norm(points-reverse,axis=2).ravel()<1.5)
                    tracked_count = int(valid.sum())
                    if tracked_count >= 12:
                        median = float(np.median(np.linalg.norm(points-tracked,axis=2).ravel()[valid])/scale)
                # Whole-frame change (0-255 grey levels) shows whether the subject also stays still.
                pairs.append(dict(frame=frame-1, pixels=median, tracked=tracked_count,
                                  change=float(cv2.absdiff(previous, gray).mean())))
            previous = gray
    finally:
        capture.release()
    return pairs


def find_stationary_interval(pairs, target_first, target_last, threshold=.5):
    runs, first = [], None
    for index in range(len(pairs)+1):
        value = pairs[index]['pixels'] if index < len(pairs) else None
        stationary = value is not None and math.isfinite(value) and value <= threshold
        if stationary and first is None:
            first = index
        if not stationary and first is not None:
            if index-first >= min(3, max(1,target_last-target_first)):
                runs.append((pairs[first]['frame'],pairs[index-1]['frame']+1))
            first = None
    candidates = [run for run in runs if max(run[0],target_first) < min(run[1],target_last)]
    if not candidates:
        raise ValueError('No reliable stationary interval overlaps the planned hold. Review the clip and enter its arrival and departure frames manually.')
    return max(candidates, key=lambda run:(min(run[1],target_last)-max(run[0],target_first),run[1]-run[0]))


def inspect_hold(source, path, keyframe):
    info = probe_video(source)
    first, last = held_path_frames(path, keyframe, info['frames'])
    return inspect_frames(source, first, last, info)


def inspect_frames(source, first, last, info=None):
    """Measured still interval overlapping the planned 0-based hold frames ``first``..``last``."""
    info = info or probe_video(source)
    pairs = background_motion(source, info)
    arrival, departure = find_stationary_interval(pairs, first, last)
    if not 0 < arrival < departure < info['frames']-1:
        raise ValueError('The detected still interval reaches a clip endpoint. Review the clip and select interior arrival/departure frames manually.')
    return dict(media=info, source_marks=[0,arrival,departure,info['frames']-1],
                target_marks=[0,first,last,info['frames']-1], motion=pairs, threshold_pixels=.5,
                note='Suggested from background motion; review the camera view before exporting. Moving backgrounds or a large subject can mislead this measurement.')


def read_generation_prompt(source):
    """Prompt WanGP stored in the MP4 comment metadata, or '' when absent."""
    result = _run([_tool('ffprobe'), '-v', 'error', '-show_entries', 'format_tags=comment', '-of', 'json', str(source)], timeout=30)
    comment = ((json.loads(result.stdout).get('format') or {}).get('tags') or {}).get('comment') or ''
    try:
        settings = json.loads(comment)
    except ValueError:
        return ''
    prompt = settings.get('prompt') if isinstance(settings, dict) else None
    return prompt if isinstance(prompt, str) else ''


def planned_hold(prompt, frames):
    """0-based (first, last) frames of the single interior hold in an H3 Camera plan, or None.

    The plan must have been compiled for this clip's frame count; several holds are left alone
    because the correction maps exactly one hold."""
    plan = PLAN_LINE.search(prompt or '')
    holds = HOLD_LINE.findall(prompt or '')
    if plan is None or int(plan[1]) != frames or len(holds) != 1:
        return None
    fps = float(plan[2])
    first, last = (math.floor(float(seconds)*fps+.5) for seconds in holds[0])
    return (first, last) if 0 < first < last < frames-1 else None


def auto_correct(source, output_dir, audio='auto', sampling='nearest'):
    """Retime a render of an H3 Camera plan so its measured hold lands on the planned frames.

    ``audio='auto'`` keeps the original soundtrack unless the plan has dialogue: stretching ambience or
    music by 2-4x smears it, while speech needs to stay in sync with the lips. Returns None when the clip
    has no single planned hold; otherwise a record whose status is 'exact', 'corrected' or 'rejected'."""
    prompt = read_generation_prompt(source)
    if PLAN_LINE.search(prompt) is None or len(HOLD_LINE.findall(prompt)) != 1:
        return None  # Not a single-hold camera plan: skip the frame-counting probe.
    info = probe_video(source)
    hold = planned_hold(prompt, info['frames'])
    if hold is None:
        return None
    if audio == 'auto':
        audio = 'retime' if DIALOGUE.search(prompt) else 'preserve'
    record = dict(source=info['path'], planned_hold_zero_based=list(hold), output=None, report=None, audio=audio)
    try:
        before = inspect_frames(info['path'], *hold, info)
        record['measured_hold_zero_based'] = before['source_marks'][1:3]
        last = info['frames']-1
        tail = settled_tail(before['motion'], before['source_marks'][2], info['frames'])
        # With retimed dialogue, shortening the ending would cut its last words; leave it.
        if tail is not None and audio != 'retime':
            # The camera settled early: end its final move on the second-to-last frame instead, unless that
            # would push the final move past the speed limits; then correct the hold alone.
            ending = dict(before, source_marks=[*before['source_marks'][:3], tail, last],
                          target_marks=[*before['target_marks'][:3], last-1, last])
            record['measured_final_arrival_zero_based'] = tail
            try:
                assess_timing(ending)
                before = ending
            except ValueError as error:
                record['ending_left_unchanged'] = str(error)
        if before['source_marks'] == before['target_marks']:
            return dict(record, status='exact')
        assess_timing(before)
        output_dir = Path(output_dir).expanduser().resolve()
        with tempfile.TemporaryDirectory(prefix='.h3-auto-', dir=output_dir) as staging:
            video, report, mapping = export_retimed(info['path'], before['source_marks'], before['target_marks'], staging, audio, sampling)
            assess_timing(inspect_frames(video, *hold), exact=True)
            published = []
            for staged in (video, report):
                destination = output_dir/Path(staged).name
                with destination.open('xb') as outgoing, open(staged, 'rb') as incoming:
                    shutil.copyfileobj(incoming, outgoing)
                published.append(str(destination))
        return dict(record, status='corrected', output=published[0], report=published[1],
                    repeated_source_frames=mapping['repeated_source_frames'])
    except (ValueError, OSError, TypeError, subprocess.SubprocessError) as error:
        return dict(record, status='rejected', reason=str(error))


def _tempo_chain(speed):
    values = []
    while speed > 2:
        values.append(2)
        speed /= 2
    while speed < .5:
        values.append(.5)
        speed /= .5
    values.append(speed)
    return ','.join(f'atempo={value:.12g}' for value in values)


def assess_timing(result, exact=False):
    """Conservative motion/timing gate; never certify viewpoint or camera roll."""
    source, target, pairs = result['source_marks'], result['target_marks'], result['motion']
    if len(pairs) != result['media']['frames']-1:
        raise ValueError('Incomplete motion evidence.')
    for index, pair in enumerate(pairs):
        if pair['frame'] != index:
            raise ValueError('Motion evidence is out of order.')
    segments = []
    for start, end in zip(source, source[1:]):
        values = [pair['pixels'] for pair in pairs[start:end]
                  if pair['pixels'] is not None and math.isfinite(pair['pixels'])]
        coverage = len(values)/(end-start)
        moving = sum(value > .5 for value in values)/max(1,len(values))
        segments.append(dict(coverage=coverage,moving_fraction=moving))
    if any(segment['coverage'] < .9 for segment in segments):
        raise ValueError('Insufficient background tracking: at least 90% coverage is required in every segment.')
    if len(segments) == 4 and (segments[3]['moving_fraction'] != 0 or not frozen_between(pairs, source[3], source[4])):
        raise ValueError('The camera must be still and the frame frozen after its final move to shorten the ending.')
    # Eased starts and far backgrounds move little at first; 40% still separates real moves from static shots.
    if any(segments[index]['moving_fraction'] < .4 for index in (0,2)):
        raise ValueError('Insufficient movement before or after the hold; a frozen or mostly static shot cannot pass.')
    if segments[1]['moving_fraction'] != 0 or source[2]-source[1] < 3:
        raise ValueError('A measured stationary hold of at least three frame intervals is required.')
    speeds = [(b-a)/(d-c) for a,b,c,d in zip(source,source[1:],target,target[1:])]
    if any(not .5 <= speed <= 2 for speed in (speeds[0], speeds[2])):
        raise ValueError('Correction would exceed the automatic 0.5x to 2x speed limit for camera movement. Review manually or try another render.')
    # The camera is measured still during the hold, so shortening or lengthening it only changes the pause.
    # Subject motion in the pause is retimed too: up to 3x in general, 10x when the whole frame is frozen.
    limit = 10 if frozen_between(pairs, source[1], source[2]) else 3
    if not 1/limit <= speeds[1] <= limit:
        raise ValueError(f'Correction would change the measured hold by more than {limit}x. Review manually or try another render.')
    # The retime maps the hold boundaries exactly; re-measuring an eased stop can land one frame either side
    # of the 0.5 px threshold, so the independent check allows one frame (1/fps) of measurement noise.
    if exact and any(abs(a-b) > 1 for a,b in zip(source,target)):
        raise ValueError('The corrected clip did not measure the planned hold boundaries within one frame.')
    return dict(segments=segments,speeds=speeds,geometry_verified=False,roll_verified=False)


def frozen_between(pairs, first, last):
    """Whether whole frames barely change from ``first`` to ``last``: neither camera nor subject moves."""
    changes = [pair.get('change') for pair in pairs[first:last] if pair.get('change') is not None]
    return bool(changes) and max(changes) < 3 and sum(changes)/len(changes) < 1


def settled_tail(pairs, after, frames, minimum=6):
    """First frame of a frozen ending reached before the last frame, or None.

    Pair ``i`` measures frames ``i`` to ``i+1``, so the camera has arrived at frame ``last moving pair + 1``."""
    moving = [pair['frame'] for pair in pairs if pair['pixels'] is not None and math.isfinite(pair['pixels']) and pair['pixels'] > .5]
    if not moving:
        return None
    arrival = moving[-1]+1
    if arrival <= after or frames-1-arrival < minimum or not frozen_between(pairs, arrival, frames-1):
        return None
    return arrival


def verified_candidates(sources, path, keyframe, output_dir, audio='retime', sampling='nearest'):
    """Try up to three existing renders; publish only after independent output checks."""
    if not sources or len(sources) > 3:
        raise ValueError('Supply one to three candidate renders in preference order.')
    if audio not in AUDIO_MODES or sampling not in SAMPLE_MODES:
        raise ValueError('Choose a supported audio mode and frame sampling method.')
    from .camera_plan import validate_path
    validate_path(path)
    _integer(keyframe,'Hold starting keyframe')
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    attempts = []
    result = dict(format='wangp-h3-camera-verified-v1',status='rejected',attempts=attempts,
                  geometry_verified=False,roll_verified=False,
                  note='Only motion and hold timing are checked. Review viewpoint, roll, subject and audio. No new video generation is launched.')
    accepted = None
    for source in sources:
        attempt = dict(source=str(source),status='rejected')
        attempts.append(attempt)
        try:
            before = inspect_hold(source,path,keyframe)
            attempt['before'] = assess_timing(before)
            attempt['source_marks'] = before['source_marks']
            attempt['target_marks'] = before['target_marks']
            # Failed exports remain private and are removed with the temporary directory.
            with tempfile.TemporaryDirectory(prefix='.h3-verified-',dir=output_dir) as staging:
                video, report, mapping = export_retimed(source,before['source_marks'],before['target_marks'],staging,audio,sampling)
                after = inspect_hold(video,path,keyframe)
                attempt['after'] = assess_timing(after,exact=True)
                attempt['measured_output_marks'] = after['source_marks']
                destination = output_dir/Path(video).name
                with destination.open('xb') as outgoing, open(video,'rb') as incoming:
                    shutil.copyfileobj(incoming,outgoing)
                accepted = str(destination)
                mapping['output'] = accepted
                result['timing'] = mapping
                attempt['status'] = 'passed_timing_checks'
                result['status'] = 'passed_timing_checks'
            break
        except (ValueError,OSError,TypeError,subprocess.SubprocessError) as error:
            attempt['reason'] = str(error)
    result['output'] = accepted
    report_path = output_dir/f'h3_camera_verification_{uuid.uuid4().hex[:10]}.json'
    with report_path.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,indent=2)
    return accepted,str(report_path),result


def audio_filter(source_marks, target_marks, fps, frame_count, sample_rate):
    # The last displayed frame occupies one more frame interval. Preserve it.
    source = [*source_marks,frame_count]
    target = [*target_marks,frame_count]
    chains, labels = [], []
    for index,(a,b,c,d) in enumerate(zip(source,source[1:],target,target[1:])):
        label = f'a{index}'
        samples = round(d/fps*sample_rate)-round(c/fps*sample_rate)
        speed = (b-a)/(d-c)
        # Tempo-shifting a long still into a few frames chirps; past 4x keep the section's start at normal speed.
        tempo = f'{_tempo_chain(speed)},' if 1/4 <= speed <= 4 else ''
        # 5 ms fades keep the joins between retimed sections from clicking.
        fade = min(.005, (d-c)/fps/4)
        fades = (f'afade=t=in:d={fade:.6f},' if index else '') + (f'areverse,afade=t=in:d={fade:.6f},areverse,'
                                                                    if index < len(source)-2 else '')
        chains.append(f'[1:a:0]atrim=start={a/fps:.12f}:end={b/fps:.12f},asetpts=PTS-STARTPTS,'
                      f'{tempo}apad,atrim=end_sample={samples},asetpts=PTS-STARTPTS,{fades}anull[{label}]')
        labels.append(f'[{label}]')
    chains.append(''.join(labels)+f'concat=n={len(labels)}:v=0:a=1[audio]')
    return ';'.join(chains)


def export_retimed(source, source_marks, target_marks, output_dir, audio='retime', sampling='nearest', progress=None):
    """Map chosen source frames to exact output frames; preserve FPS and duration."""
    import cv2
    info = probe_video(source)
    mapping = frame_map(info['frames'], source_marks, target_marks)
    if audio not in AUDIO_MODES or sampling not in SAMPLE_MODES:
        raise ValueError('Choose a supported audio mode and frame sampling method.')
    source_marks = [_integer(value,'Source mark') for value in source_marks]
    target_marks = [_integer(value,'Target mark') for value in target_marks]
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    stem = Path(info['path']).stem[:48].rstrip('. ') or 'camera'
    destination = output_dir/f'{stem}_timed_{uuid.uuid4().hex[:10]}.mp4'
    if destination.exists():
        raise ValueError('Output already exists; choose another export.')
    sampled = [math.floor(position+Fraction(1,2)) for position in mapping]
    encoder = None
    capture = cv2.VideoCapture(info['path'])
    with tempfile.TemporaryDirectory(prefix='.h3-camera-',dir=output_dir) as temporary:
        video = Path(temporary)/'video.mp4'
        finished = Path(temporary)/'finished.mp4'
        with tempfile.TemporaryFile() as errors:
            try:
                command = [_tool('ffmpeg'),'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24',
                           '-s',f"{info['width']}x{info['height']}",'-r',info['fps_fraction'],'-i','pipe:0',
                           '-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
                           '-frames:v',str(info['frames']),'-movflags','+faststart',str(video)]
                encoder = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=errors,
                                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                decoded, cache = -1, {}
                for frame, position in enumerate(mapping):
                    low = math.floor(position) if sampling == 'blend' else sampled[frame]
                    high = math.ceil(position) if sampling == 'blend' else low
                    while decoded < high:
                        ok, pixels = capture.read()
                        decoded += 1
                        if not ok:
                            raise ValueError(f'Could not decode source frame {decoded+1}.')
                        cache[decoded] = pixels
                    pixels = cache[low]
                    if high != low:
                        amount = float(position-low)
                        pixels = cv2.addWeighted(pixels,1-amount,cache[high],amount,0)
                    encoder.stdin.write(pixels.tobytes())
                    cache = {index:pixels for index,pixels in cache.items() if index >= low}
                    if progress and frame % 12 == 0:
                        progress(frame/info['frames'])
                encoder.stdin.close()
                if encoder.wait(timeout=90):
                    errors.seek(0)
                    raise ValueError('Video export failed: '+errors.read().decode('utf-8',errors='replace')[-1000:])
            except BrokenPipeError as error:
                if encoder is not None:
                    encoder.wait(timeout=10)
                errors.seek(0)
                raise ValueError('Video export failed: '+errors.read().decode('utf-8',errors='replace')[-1000:]) from error
            finally:
                capture.release()
                if encoder is not None and encoder.poll() is None:
                    encoder.kill()
                    encoder.wait(timeout=10)
        command = [_tool('ffmpeg'),'-v','error','-n','-i',str(video)]
        if info['has_audio'] and audio != 'mute':
            command += ['-i',info['path']]
            if audio == 'retime':
                command += ['-filter_complex',audio_filter(source_marks,target_marks,info['fps'],info['frames'],info['audio_sample_rate']),'-map','0:v:0','-map','[audio]']
            else:
                command += ['-map','0:v:0','-map','1:a:0','-af',f"apad,atrim=duration={info['duration']:.12f}"]
            command += ['-c:a','aac','-b:a','192k']
        else:
            command += ['-map','0:v:0','-an']
        command += ['-c:v','copy','-map_metadata','-1','-t',f"{info['duration']:.12f}",'-movflags','+faststart',str(finished)]
        _run(command,timeout=120)
        measured = probe_video(finished)
        if any(measured[key] != info[key] for key in ('frames','fps_fraction','width','height')):
            raise ValueError('Export verification failed: frame count, FPS or dimensions changed.')
        _run([_tool('ffmpeg'),'-v','error','-i',str(finished),'-map','0:v:0','-map','0:a?','-f','null','-'],timeout=60)
        with destination.open('xb') as output, finished.open('rb') as incoming:
            shutil.copyfileobj(incoming,output)
    record = dict(format='wangp-h3-camera-timing-v1', source=info,
                  source_sha256=hashlib.sha256(Path(info['path']).read_bytes()).hexdigest(), output=str(destination),
                  source_marks_zero_based=source_marks,target_marks_zero_based=target_marks,
                  source_frame_for_output=[float(value) for value in mapping],
                  exact_boundaries=[dict(source_frame=a+1,output_frame=b+1,source_seconds=a/info['fps'],output_seconds=b/info['fps']) for a,b in zip(source_marks,target_marks)],
                  sampling=sampling,audio_mode=audio if info['has_audio'] else 'no source audio',
                  repeated_source_frames=len(sampled)-len(set(sampled)),
                  dropped_source_frames=info['frames']-len(set(sampled)),
                  full_decode='passed',
                  note='Selected frame boundaries are exact. Camera geometry and correctness of the selected views are not inferred. Retiming also changes subject motion; nearest sampling can repeat frames, while blending can ghost.')
    report = destination.with_suffix('.timing.json')
    with report.open('x',encoding='utf-8') as handle:
        json.dump(record,handle,indent=2)
    if progress:
        progress(1)
    return str(destination),str(report),record
