import RecordRTC, {StereoAudioRecorder} from "recordrtc";
import hark from "hark";

import {v4 as uuidv4} from "uuid";

import {configureStore} from "@reduxjs/toolkit";

import mediaReducer from "./reducers/media";

export function setupStore() {
    return configureStore({
        reducer: {
            media: mediaReducer,
        },
    });
}

function setupStream() {
    if (window.audioStream) {
        return Promise.resolve(window.audioStream);
    } else {
        return navigator.mediaDevices.getUserMedia({video: false, audio: true}).then(stream => {
            window.audioStream = stream;
            return stream;
        });
    }
}

// --------NEW CLASS: InterruptionRecordRTC--------
class InterruptionRecordRTC {
    constructor(stream, options, uuid) {
        this.recorder = RecordRTC(stream, options);
        this.interruptUuid = uuid;
    }

    getSampleRate() {
        return this.recorder.sampleRate;
    }

    getBufferSize() {
        return this.recorder.bufferSize;
    }

    getBlob() {
        return this.recorder.getBlob();
    }

    startRecording() {
        return this.recorder.startRecording();
    }

    stopRecording(callback) {
        return this.recorder.stopRecording(callback);
    }

    getDataURL(callback) {
        return this.recorder.getDataURL(callback);
    }

    destroy() {
        return this.recorder.destroy();
    }

    getInterruptId() {
        return this.interruptUuid;
    }
}

// --------NEW FUNCTION: createInterruptionRecorder--------
// factory method, so we can instantiate similarly the RecordRTC
function createInterruptionRecorder(stream, options, uuid) {
    return new InterruptionRecordRTC(stream, options, uuid);
}

export function setupAudioRecorder(checkInterruption = false, queryInterruption = null) { 
    // --------START OF NEW CODE SECTION--------
    let options = {
        recorderType: StereoAudioRecorder,
        type: "audio",
        mimeType: "audio/wav",
        numberOfAudioChannels: 1,
        desiredSampRate: 16000,
    };
    if (checkInterruption) {
        let uuid = uuidv4(); // all interruptions at intervals share the same id
        options.timeSlice = 1000 // concatenate intervals based blobs                             
        options.ondataavailable = async function(blob) {        
            let reader = new window.FileReader();
            reader.readAsDataURL(blob);
            reader.onloadend = function() {
                var audioBase64 = reader.result;
                let audioTurned = audioBase64.substr(audioBase64.indexOf(',') + 1);
                queryInterruption({
                  id: uuid,
                  audio: {
                    type: blob.type || "audio/wav",
                    sampleRate: 16000,
                    bufferSize: null,
                    data: audioTurned
                  }
                });
                console.log('CALL MADE TO INTERRUPTION SERVICE')
            };
        }
        return setupStream().then(stream =>
            createInterruptionRecorder(stream, options, uuid)
        );
    } 
    // --------END OF NEW CODE SECTION--------
    else {
        return setupStream().then(stream =>
            RecordRTC(stream, options)
        );
    }
}

export function setupHark() {
    return setupStream().then(stream => hark(stream, {play: false}));
}

export function replaceHtmlElements(text) {
    return text.replace(/\n/gi, "<br />");
}