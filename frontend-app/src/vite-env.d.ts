/// <reference types="vite/client" />

/** Set by vite.config.ts: this build's id, carried by its API reads. */
declare const __BUILD_ID__: string;

declare module '*.mp4' {
    const src: string;
    export default src;
}
