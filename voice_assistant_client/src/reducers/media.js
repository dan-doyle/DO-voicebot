import {createSlice} from "@reduxjs/toolkit";
import {PlayerStatus} from "./const";

export const media = createSlice({
    name: "media",
    initialState: {
        status: PlayerStatus.IDLE,
        responseList: [],
        audio: null,
        receivedHotwordRes: false,
        detectedHotword: false,
        isInterrupt: { // NEW 
            value: false,
            id: null,
        }
    },
    reducers: {
        changeStatus: (state, action) => {
            state.status = action.payload.status;
        },
        addResponse: (state, action) => {
            state.responseList = [{
                id: action.payload.id,
                date: action.payload.date,
                command: action.payload.command,
                emotion: action.payload.emotion,
                human_thayers: action.payload.human_thayers,
                bot_thayers: action.payload.bot_thayers,
                grammar_positions: action.payload.grammar_positions,
                grammar_prediction: action.payload.grammar_prediction,
                response: action.payload.response,
                error: action.payload.error
            }, ...state.responseList];
            state.audio = createDataUrl(action.payload.audio);
            if (!state.audio) {
                state.status = PlayerStatus.IDLE;
            }
        },
        hotwordResponse: (state, action) => {
            state.receivedHotwordRes = action.payload.value
        },
        foundHotword: (state, action) => {
            state.detectedHotword = action.payload.value
        },
        clearAudio: (state) => {
            state.status = PlayerStatus.IDLE;
            state.audio = null;
        },
        // --------NEW REDUCER: clearAudioPostInterruption--------
        clearAudioPostInterruption: (state) => {
            state.audio = null;
        },
        // --------NEW REDUCER: userInterruptDetected--------
        userInterruptDetected: (state, action) => { // changes state isInterrupt which in turn pauses audio in SimpleRecorder
            if(state.status == PlayerStatus.RESPONDING) {
                state.status = PlayerStatus.LISTENING;
            } else {
                state.status = PlayerStatus.IDLE;
            }
            state.isInterrupt = {value: action.payload.value, id: action.payload.id};
        },
    },
});

export const {changeStatus, addResponse, hotwordResponse, foundHotword, clearAudio, clearAudioPostInterruption, userInterruptDetected } = media.actions;

export const selectStatus = state => state.media.status;
export const selectResponses = state => state.media.responseList;
export const selectAudio = state => state.media.audio;
export const selectInterrupt = state => state.media.isInterrupt; // NEW

export default media.reducer;

// utilities
function createDataUrl(audio) {
    return (audio && audio.data) ? `data:${audio.contentType};base64,${audio.data}` : null;
}