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

export function setupAudioRecorder(checkInterruption = false, queryInterruption = null) { // dependency invert: , ondataavailableCallback = null
    if (checkInterruption) {
        let uuid = uuidv4() // all interruptions at intervals share the same id
        return setupStream().then(stream =>
            RecordRTC(stream, {
                recorderType: StereoAudioRecorder,
                type: "audio",
                mimeType: "audio/wav",
                numberOfAudioChannels: 1,
                desiredSampRate: 16000,
                timeSlice: 1000, // concatenate intervals based blobs
                // if interruption is found, we don't need to make calls from 'ondataavailable' callback
                // ondataavailable: async function(blob) {
                //     // extract out part of function to emit to socket
                //     console.log('ondataavailable CALLED')
                //     console.log(blob)
                //     // let audioDataURL = URL.createObjectURL(blob)
                //     queryInterruption({ // try and not have these hard-coded
                //         audio: {
                //             type: blob.type || "audio/wav",
                //             sampleRate: 16000,
                //             bufferSize: null, 
                //             data: blob //audioDataURL.split(",").pop()
                //         }
                //     });
                // }
                ondataavailable: async function(blob) {
                    // console.log('CALLED ondataavailable');
                    // console.log(blob);
                    // let audioURL = URL.createObjectURL(blob);
                    // const a = document.createElement('a');
                    // a.href = audioURL;
                    // a.download = 'NewInterruptAudio.wav'; // You can name the file whatever you want
                    // a.textContent = 'Download the audio file';
                    // document.body.appendChild(a);
                    // a.click();

                  
                    // const arrayBuffer = await blob.arrayBuffer(); // Didn't work
                    // const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
                    
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
            })
        );
    } else {
        return setupStream().then(stream =>
            RecordRTC(stream, {
                recorderType: StereoAudioRecorder,
                type: "audio",
                mimeType: "audio/wav",
                numberOfAudioChannels: 1,
                desiredSampRate: 16000,
            })
        );
    }
}

export function setupHark() {
    return setupStream().then(stream => hark(stream, {play: false}));
}

export function replaceHtmlElements(text) {
    return text.replace(/\n/gi, "<br />");
}