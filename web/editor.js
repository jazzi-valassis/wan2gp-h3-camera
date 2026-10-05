(() => {
    "use strict";
    const namespace = "wan2gp:h3-camera:v1";
    const $ = id => document.getElementById(id);
    const origin = () => ({time:0, azimuth:0, elevation:0, distance:1});
    let points = [origin(), {time:1, azimuth:90, elevation:0, distance:1}];
    let selected = 1, position = 1, seconds = 5, timingKnown = false, easing = "smooth";
    let invalid = false, connected = false, playing = false, animation = 0, startTime = 0;
    let drag = null, lastPublished = "", lastHeight = 0;
    const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
    const round = value => Math.round(value * 1000000) / 1000000;
    const post = (type, value) => parent.postMessage({namespace, type, value}, "*");
    const fmt = value => Number(value.toFixed(2)).toString();
    const signed = value => (value > 0 ? "+" : "") + fmt(value);
    // Orbit angles are absolute; a segment's turn is its change from the previous keyframe.
    const turnOf = index => index ? points[index].azimuth - points[index - 1].azimuth : 0;
    const validPoint = p => p && ["time", "azimuth", "elevation", "distance"].every(k => typeof p[k] === "number" && Number.isFinite(p[k]));

    function parsePath(value) {
        if (new TextEncoder().encode(value).length > 32768) throw Error("The path must be smaller than 32 KB.");
        const parsed = JSON.parse(value);
        if (!Array.isArray(parsed) || parsed.length < 2 || parsed.length > 24) throw Error("Use 2 to 24 keyframes.");
        let travel = 0;
        parsed.forEach((p, i) => {
            if (!validPoint(p)) throw Error("Each keyframe needs numeric time, azimuth, elevation and distance values.");
            if (Object.keys(p).sort().join(",") !== "azimuth,distance,elevation,time") throw Error("Use only time, azimuth, elevation and distance in each keyframe.");
            if (p.time < 0 || p.time > 1 || (i && p.time <= parsed[i - 1].time)) throw Error("Keyframe times must increase from 0 to 1.");
            if (Math.abs(p.azimuth) > 11520 || Math.abs(p.elevation) > 89 || p.distance < .1 || p.distance > 4) throw Error("Camera position is outside the supported range.");
            if (i) travel += Math.abs(p.azimuth - parsed[i - 1].azimuth);
        });
        const first = parsed[0];
        if (first.time !== 0 || first.azimuth !== 0 || first.elevation !== 0 || first.distance !== 1 || parsed.at(-1).time !== 1) throw Error("Keep the start at (0, 0°, 0°, 1×) and the last time at 1.");
        if (travel > 11520) throw Error("Total camera rotation cannot exceed 32 turns.");
        return parsed.map(p => ({time:p.time, azimuth:p.azimuth, elevation:p.elevation, distance:p.distance}));
    }

    function resize() {
        const height = Math.ceil($("editor").getBoundingClientRect().height + 2);
        if (height !== lastHeight) { lastHeight = height; post("resize", height); }
    }
    function showError(message = "") {
        $("error").textContent = message;
        $("error").hidden = !message;
        resize();
    }
    function stop() {
        playing = false;
        cancelAnimationFrame(animation);
        $("play").textContent = "Play preview";
        $("play").setAttribute("aria-pressed", "false");
    }
    function publish() {
        lastPublished = JSON.stringify(points);
        post("edit", lastPublished);
        $("connection").textContent = connected ? "Saving camera path…" : "Waiting for the generation form. Changes are not applied yet.";
    }
    function edited() {
        stop();
        position = points[selected].time;
        $("preset").value = "custom";
        showError();
        render();
        publish();
    }
    function at(time) {
        let right = points.findIndex(p => p.time >= time);
        if (right <= 0) return {...points[0]};
        const a = points[right - 1], b = points[right];
        let amount = clamp((time - a.time) / (b.time - a.time), 0, 1);
        if (easing === "smooth") amount = amount * amount * (3 - 2 * amount);
        return {time, azimuth:a.azimuth + (b.azimuth - a.azimuth) * amount,
                elevation:a.elevation + (b.elevation - a.elevation) * amount,
                distance:a.distance + (b.distance - a.distance) * amount};
    }
    function timeLabel(time) { return timingKnown ? `${fmt(time * seconds)} s` : `${fmt(time * 100)}%`; }
    function updatePosition() {
        $("scrub").value = Math.round(position * 1000);
        $("scrub-time").value = `${timeLabel(position)} / ${timeLabel(1)}`;
        draw();
    }
    function draw() {
        const rad = Math.PI / 180, cx = 228, cy = 155;
        const largest = Math.max(1, ...points.map(p => p.distance));
        const scale = 118 / largest;
        const xyz = p => {
            const el = p.elevation * rad, az = p.azimuth * rad;
            return {x:Math.sin(az) * Math.cos(el) * p.distance,
                    y:Math.sin(el) * p.distance,
                    z:Math.cos(az) * Math.cos(el) * p.distance};
        };
        const project = ({x, y = 0, z}) => ({x:cx + (x + z * .35) * scale, y:cy + (z * .38 - y * .87) * scale});
        const point = p => project(xyz(p));
        const xy = p => `${p.x.toFixed(2)},${p.y.toFixed(2)}`;
        const grid = [];
        for (let n = -2; n <= 2; n++) {
            const v = n * largest / 2;
            grid.push(`<path class="grid-line" d="M${xy(project({x:-largest,z:v}))} L${xy(project({x:largest,z:v}))} M${xy(project({x:v,z:-largest}))} L${xy(project({x:v,z:largest}))}"/>`);
        }
        const ring = [];
        for (let a = 0; a <= 360; a += 5) ring.push(xy(point({azimuth:a,elevation:0,distance:1})));
        grid.push(`<polyline class="grid-line" points="${ring.join(" ")}"/>`);
        const path = [];
        points.slice(1).forEach((b, i) => {
            const a = points[i];
            const steps = Math.min(256, Math.max(16, Math.ceil(Math.abs(b.azimuth - a.azimuth) / 5)));
            for (let n = i ? 1 : 0; n <= steps; n++) path.push(xy(point(at(a.time + (b.time - a.time) * n / steps))));
        });
        const active = point(at(position)), start = point(points[0]);
        const markers = points.map((p, i) => {
            if (!i) return "";
            const pos = point(p);
            return `<g data-keyframe="${i}"><circle class="point${selected === i ? " selected" : ""}" cx="${pos.x}" cy="${pos.y}" r="7"/><text class="frame-number" x="${pos.x+10}" y="${pos.y-9}">${i + 1}</text></g>`;
        }).join("");
        $("scene").innerHTML = `${grid.join("")}<polyline class="path-line" points="${path.join(" ")}"/>
            <path class="sight-line" d="M${xy(active)} L${cx},${cy-12}"/>
            <ellipse class="subject" cx="${cx}" cy="${cy+2}" rx="18" ry="7"/>
            <path class="subject" d="M${cx-12},${cy} L${cx-10},${cy-24} Q${cx},${cy-35} ${cx+10},${cy-24} L${cx+12},${cy} Z"/>
            <circle class="subject" cx="${cx}" cy="${cy-41}" r="9"/>
            <text class="diagram-label" x="${cx}" y="${cy+26}" text-anchor="middle">Subject</text>
            <circle class="origin-point" cx="${start.x}" cy="${start.y}" r="6"/>
            ${markers}<g transform="translate(${active.x},${active.y})" pointer-events="none"><rect class="camera-icon" x="-10" y="-7" width="15" height="13" rx="2"/><path class="camera-icon" d="M5,-3 L13,-7 L13,6 L5,2Z"/></g>
            <text class="diagram-label" x="15" y="23">${fmt(at(position).azimuth)}° orbit from start · ${fmt(at(position).elevation)}° elevation · ${fmt(at(position).distance)}×</text>`;
    }
    function segmentHelp(index) {
        const turn = turnOf(index);
        const prior = index > 1 ? turnOf(index - 1) : 0;
        const heading = points.slice(1, index).map((_, i) => turnOf(i + 1)).filter(Boolean).at(-1) || 0;
        if (!turn) return `Same orbit angle as keyframe ${index}: ${prior ? "the orbit stops" : "no sideways orbit"} in this segment.`;
        const reverse = heading && (turn > 0) !== (heading > 0) ? ", reversing the previous direction" : "";
        return `This segment orbits ${fmt(Math.abs(turn))}° toward camera ${turn > 0 ? "right" : "left"}${reverse}.`;
    }
    function render() {
        const p = points[selected], fixed = selected === 0;
        $("selected-title").textContent = `Keyframe ${selected + 1}${fixed ? " · Start" : ""}`;
        $("selected-time").textContent = timeLabel(p.time);
        $("time").value = round(p.time * 100);
        $("time").disabled = invalid || fixed || selected === points.length - 1;
        $("time").min = selected ? round((points[selected - 1].time + .00001) * 100) : 0;
        $("time").max = selected < points.length - 1 ? round((points[selected + 1].time - .00001) * 100) : 100;
        ["azimuth", "elevation", "distance"].forEach(key => { $(key).value = fmt(p[key]); $(key).disabled = invalid || fixed; });
        $("turn").value = fmt(turnOf(selected));
        $("turn").disabled = invalid || fixed;
        $("distance-range").value = p.distance;
        $("distance-range").disabled = invalid || fixed;
        $("keyframe-help").textContent = fixed ? "The start view is fixed. Select another keyframe to move the camera." : segmentHelp(selected);
        $("add").disabled = invalid || points.length >= 24;
        $("remove").disabled = invalid || fixed || points.length <= 2;
        $("play").disabled = invalid;
        $("scrub").disabled = invalid;
        $("frame-count").textContent = `${points.length} / 24 keyframes`;
        const focusedFrame = document.activeElement?.dataset.keyframe;
        $("keyframes").innerHTML = points.map((point, i) => `<button type="button" class="keyframe" data-keyframe="${i}" aria-pressed="${i === selected}" aria-label="Select keyframe ${i+1} at ${timeLabel(point.time)}, orbit angle ${fmt(point.azimuth)} degrees${i ? `, turn ${signed(turnOf(i))} degrees` : ""}"${invalid ? " disabled" : ""}>${i === 0 ? "Start" : `Keyframe ${i + 1}`}<span>${timeLabel(point.time)} · ${fmt(point.azimuth)}°${i ? ` (${signed(turnOf(i))})` : ""}</span></button>`).join("");
        if (focusedFrame !== undefined) $("keyframes").querySelector(`[data-keyframe="${focusedFrame}"]`)?.focus({preventScroll:true});
        updatePosition();
        resize();
    }
    function select(index) {
        if (invalid) return;
        stop();
        selected = clamp(index, 0, points.length - 1);
        position = points[selected].time;
        render();
    }
    function setCoordinate(key, value) {
        if (invalid || !selected || !Number.isFinite(value)) { render(); return; }
        const limits = {azimuth:[-11520,11520], elevation:[-89,89], distance:[.1,4]};
        const previous = points[selected][key];
        points[selected][key] = round(clamp(value, ...limits[key]));
        try { parsePath(JSON.stringify(points)); } catch (error) {
            points[selected][key] = previous;
            showError(error.message);
            return;
        }
        edited();
    }
    function setTurn(value) {
        if (invalid || !selected || !Number.isFinite(value)) { render(); return; }
        // Shift this and every later keyframe, so later segments keep their own turns.
        const shift = round(points[selected - 1].azimuth + value - points[selected].azimuth);
        const previous = points.map(p => p.azimuth);
        points.slice(selected).forEach(p => { p.azimuth = round(p.azimuth + shift); });
        try { parsePath(JSON.stringify(points)); } catch (error) {
            points.forEach((p, i) => { p.azimuth = previous[i]; });
            render();
            showError(error.message);
            return;
        }
        edited();
    }
    ["azimuth", "elevation", "distance"].forEach(key => $(key).addEventListener("change", e => setCoordinate(key, e.target.valueAsNumber)));
    $("turn").addEventListener("change", e => setTurn(e.target.valueAsNumber));
    $("distance-range").addEventListener("input", e => setCoordinate("distance", Number(e.target.value)));
    $("time").addEventListener("change", e => {
        if (invalid || selected === 0 || selected === points.length - 1) return;
        const value = e.target.valueAsNumber / 100;
        if (!Number.isFinite(value)) { render(); return; }
        const lower = points[selected - 1].time, upper = points[selected + 1].time;
        const margin = Math.min(.00001, (upper - lower) / 4);
        const next = clamp(value, lower + margin, upper - margin);
        if (next <= lower || next >= upper) { render(); showError("These keyframes are too close together to move between them."); return; }
        points[selected].time = next;
        edited();
    });
    $("keyframes").addEventListener("click", e => { const button = e.target.closest("[data-keyframe]"); if (button) select(Number(button.dataset.keyframe)); });
    $("add").addEventListener("click", () => {
        if (invalid || points.length >= 24) return;
        let insert = Math.min(selected + 1, points.length - 1);
        if (points[insert].time - points[insert - 1].time < .00002) {
            insert = points.slice(1).reduce((best, p, i) => p.time-points[i].time > points[best].time-points[best-1].time ? i+1 : best, 1);
        }
        const time = round((points[insert - 1].time + points[insert].time) / 2);
        points.splice(insert, 0, at(time));
        selected = insert;
        edited();
    });
    $("remove").addEventListener("click", () => {
        if (invalid || selected === 0 || points.length <= 2) return;
        points.splice(selected, 1);
        points[points.length - 1].time = 1;
        selected = Math.min(selected, points.length - 1);
        edited();
    });
    $("preset").addEventListener("change", e => {
        const type = e.target.value;
        if (type === "custom") return;
        const last = {...origin(),time:1};
        const rotations = {right90:90,left90:-90,half:180,full:360};
        if (rotations[type]) last.azimuth = rotations[type];
        if (type === "push") last.distance = .65;
        if (type === "pull") last.distance = 1.5;
        if (type === "rise") last.elevation = 20;
        points = [origin(), last]; selected = 1; invalid = false;
        edited();
        $("preset").value = type;
    });
    $("scrub").addEventListener("input", e => { stop(); position = Number(e.target.value) / 1000; updatePosition(); });
    function tick(now) {
        if (!playing) return;
        position = clamp((now - startTime) / (seconds * 1000), 0, 1);
        updatePosition();
        if (position >= 1) stop(); else animation = requestAnimationFrame(tick);
    }
    $("play").addEventListener("click", () => {
        if (playing) { stop(); return; }
        if (position >= 1) position = 0;
        playing = true;
        startTime = performance.now() - position * seconds * 1000;
        $("play").textContent = "Pause preview";
        $("play").setAttribute("aria-pressed", "true");
        animation = requestAnimationFrame(tick);
    });
    $("orbit").addEventListener("pointerdown", e => {
        if (e.button !== 0 || invalid) return;
        const marker = e.target.closest("[data-keyframe]");
        if (marker) select(Number(marker.dataset.keyframe));
        $("orbit").focus({preventScroll:true});
        if (!selected) return;
        stop();
        drag = {x:e.clientX,y:e.clientY,azimuth:points[selected].azimuth,elevation:points[selected].elevation,axis:null};
        $("orbit").setPointerCapture(e.pointerId);
        $("orbit").classList.add("dragging");
        e.preventDefault();
    });
    $("orbit").addEventListener("pointermove", e => {
        if (!drag) return;
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        if (!drag.axis) {
            if (Math.max(Math.abs(dx), Math.abs(dy)) < 3) return;
            drag.axis = Math.abs(dx) >= Math.abs(dy) ? "azimuth" : "elevation";
        }
        const old = {...points[selected]};
        if (drag.axis === "azimuth") points[selected].azimuth = round(clamp(drag.azimuth + dx * .7, -11520, 11520));
        if (drag.axis === "elevation") points[selected].elevation = round(clamp(drag.elevation - dy * .5, -89, 89));
        try { parsePath(JSON.stringify(points)); } catch (_) { points[selected] = old; return; }
        edited();
    });
    for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) $("orbit").addEventListener(event, () => { drag = null; $("orbit").classList.remove("dragging"); });
    $("orbit").addEventListener("wheel", e => {
        // Only consume wheel gestures while the diagram has focus; page scrolling remains available.
        if (document.activeElement !== $("orbit") || invalid || !selected) return;
        e.preventDefault();
        setCoordinate("distance", points[selected].distance + Math.sign(e.deltaY) * .05);
    }, {passive:false});
    $("orbit").addEventListener("keydown", e => {
        if (invalid) return;
        const delta = e.shiftKey ? 10 : 1;
        if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) {
            e.preventDefault();
            const key = ["ArrowLeft", "ArrowRight"].includes(e.key) ? "azimuth" : "elevation";
            const direction = ["ArrowLeft", "ArrowDown"].includes(e.key) ? -1 : 1;
            setCoordinate(key, points[selected][key] + direction * delta);
        }
        if (e.key === "Escape") { stop(); drag = null; }
    });
    window.addEventListener("message", e => {
        if (e.source !== parent || e.data?.namespace !== namespace) return;
        const {type,value} = e.data;
        if (type === "path" && typeof value === "string") {
            if (value === lastPublished && !invalid) return;
            try {
                const parsed = parsePath(value);
                if (JSON.stringify(parsed) !== JSON.stringify(points) || invalid) {
                    stop(); points = parsed; selected = Math.min(selected, points.length - 1); position = points[selected].time;
                    invalid = false; $("preset").value = "custom"; showError(); render();
                }
            } catch (error) { invalid = true; stop(); showError(`Path JSON: ${error.message} Choose a camera move to reset it, or fix the JSON below.`); render(); }
        }
        if (type === "timing") {
            try {
                const timing = typeof value === "string" ? JSON.parse(value) : value;
                const next = Number(timing.seconds);
                if (Number.isFinite(next) && next > 0) {
                    seconds = next; timingKnown = true;
                    easing = timing.easing === "linear" ? "linear" : "smooth";
                    $("duration").textContent = `Last frame ${fmt(seconds)} s · ${Number(timing.frame_count)} frames · ${fmt(Number(timing.fps))} fps`;
                    if (playing) startTime = performance.now() - position * seconds * 1000;
                    render();
                }
            } catch (_) { /* Keep last valid timing until the form has finished updating. */ }
        }
        if (type === "theme") {
            document.documentElement.classList.toggle("dark", !!value?.dark);
            if (typeof value?.font === "string") document.documentElement.style.setProperty("--host-font", value.font);
            for (const name of ["surface", "panel", "text", "muted", "line", "accent"]) {
                const color = value?.colors?.[name];
                if (typeof color === "string" && CSS.supports("color", color)) document.documentElement.style.setProperty(`--${name}`, color);
            }
        }
        if (type === "connected") {
            connected = !!value;
            if (!connected) $("connection").textContent = "Waiting for the generation form. Changes are not applied yet.";
            else if (!lastPublished) $("connection").textContent = "Camera path connected to the generation form.";
        }
        if (type === "saved" && value === lastPublished) $("connection").textContent = "Camera path updated. Use the button below to apply its prompt.";
    });
    document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
    new ResizeObserver(resize).observe($("editor"));
    render();
    post("ready");
})();
