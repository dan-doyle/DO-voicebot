import io from "socket.io-client";

import {v4 as uuidv4} from "uuid";

import {addResponse, clearAudio, hotwordResponse, foundHotword, changeStatus, userInterruptDetected} from "./media";
import {PlayerStatus} from "./const";

export function setupSocket(backendUrl, dispatch) {
    window.socket = io(backendUrl);

    window.socket.on("connect", () => console.log("Socket connected ..."));
    window.socket.on("response", (data) => {
        console.log("received", data);
        
        dispatch(addResponse({
            ...data,
            id: data.id || uuidv4(),
            date: data.date || new Date(),
            audio: {
                data: data.audio.data,
                contentType: data.audio.contentType
            }
        }));
    });
    window.socket.on("received-hotword-response", () => {
        dispatch(hotwordResponse({value: true}))
    });
    window.socket.on("hotword", () => {
        dispatch(foundHotword({value: true}))
    });
    // new: causes the pausing of playback
    window.socket.on("interrupt-detected", (data) => {
        console.log("USER INTERRUPTION DETECTED")
        dispatch(userInterruptDetected({value: true, id: data.id}))
    });
    window.socket.on("disconnect", () => console.log("Socket disconnected ..."));
}

export function submitRecording(payload) {
    return dispatch => {
        window.socket.emit("audio-command", {
            type: "audio",
            ...payload,
            id: payload.id || uuidv4(),
            date: payload.date || new Date()
        });
        // temporary start
        // var new_payload = {command: 'Fake audio message'}
        // window.socket.emit("text-command", {
        //     type: "command",
        //     ...new_payload,
        //     id: payload.id || uuidv4(),
        //     date: payload.date || new Date()
        // });
        // temporary end
        dispatch(changeStatus({status: PlayerStatus.PROCESSING}));
    };
}

export function submitHotwordRecording(payload) {
    return window.socket.emit("audio-hotword", {
        type: "audio",
        ...payload,
        id: payload.id || uuidv4(),
        date: payload.date || new Date()
    });
}

export function submitCommand(payload) {
    return dispatch => {
        window.socket.emit("text-command", {
            type: "command",
            ...payload,
            id: payload.id || uuidv4(),
            date: payload.date || new Date()
        });
        dispatch(changeStatus({status: PlayerStatus.PROCESSING}));
    };
}

/* Create a function to emit the "audio-interrupt" event in the VA Service
    - Should call userInterrupt from redux
*/
// should move this function out of redux, unless we want to introduce some 'queryInterruption' state
export function queryInterruption(payload) { // introduce payload param
    console.log('QUERYING VA SERVICE TO CHECK FOR INTERRUPTION')
    window.socket.emit("audio-interrupt", {
        type: "command",
        ...payload,
        date: payload.date || new Date()
    });
}