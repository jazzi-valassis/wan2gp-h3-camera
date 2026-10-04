() => {
    const namespace = "wan2gp:h3-camera:v1";
    if (window.__wan2gpH3CameraBridge) {
        window.__wan2gpH3CameraBridge.refresh();
        return;
    }
    let previousFrame, lastPath, lastTiming, lastTheme;
    const findFrame = () => document.querySelector("iframe#h3-camera-editor");
    const field = id => document.querySelector(`#${id} textarea, #${id} input`);
    const send = (frame, type, value) => frame?.contentWindow?.postMessage({namespace, type, value}, "*");
    const refresh = (force = true) => {
        const frame = findFrame();
        if (!frame) return;
        if (previousFrame !== frame) {
            previousFrame = frame;
            lastPath = lastTiming = lastTheme = undefined;
            frame.addEventListener("load", () => refresh(true));
        }
        const path = field("h3-camera-path");
        const timing = field("h3-camera-timing");
        const host = document.querySelector(".gradio-container") || document.body;
        const style = getComputedStyle(host);
        const colors = {};
        const variables = {surface:"--block-background-fill", panel:"--background-fill-secondary", text:"--body-text-color", muted:"--body-text-color-subdued", line:"--border-color-primary", accent:"--color-accent"};
        for (const [name, variable] of Object.entries(variables)) {
            const color = style.getPropertyValue(variable).trim();
            if (color && CSS.supports("color", color)) colors[name] = color;
        }
        const theme = JSON.stringify({dark:!!document.querySelector(".dark"),font:style.fontFamily,colors});
        if (path && (force || path.value !== lastPath)) {
            lastPath = path.value;
            send(frame, "path", lastPath);
        }
        if (timing && (force || timing.value !== lastTiming)) {
            lastTiming = timing.value;
            send(frame, "timing", lastTiming);
        }
        if (force || theme !== lastTheme) {
            lastTheme = theme;
            send(frame, "theme", JSON.parse(theme));
        }
        send(frame, "connected", !!path);
    };
    window.addEventListener("message", event => {
        const frame = findFrame();
        if (!frame || event.source !== frame.contentWindow || event.data?.namespace !== namespace) return;
        const {type, value} = event.data;
        if (type === "ready") refresh(true);
        if (type === "resize" && Number.isFinite(value)) frame.style.height = `${Math.max(400, Math.min(1600, Math.ceil(value)))}px`;
        if (type !== "edit" || typeof value !== "string" || value.length > 65536) return;
        const target = field("h3-camera-path");
        if (!target) { send(frame, "connected", false); return; }
        const prototype = target.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
        Object.getOwnPropertyDescriptor(prototype, "value").set.call(target, value);
        lastPath = value;
        target.dispatchEvent(new Event("input", {bubbles:true}));
        send(frame, "saved", value);
    });
    const timer = setInterval(() => refresh(false), 180);
    window.__wan2gpH3CameraBridge = {refresh, timer};
    refresh(true);
}
