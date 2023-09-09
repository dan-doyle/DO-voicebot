import React, { useRef, useEffect } from "react"; // hooks are new
import {useDispatch, useSelector} from "react-redux";

import ReactAudioPlayer from "react-audio-player";

import {changeStatus, clearAudio, clearAudioPostInterruption, selectAudio, selectInterrupt} from "../reducers/media"; // new userInterrupt
import {PlayerStatus} from "../reducers/const";


const SimpleRecorder = () => {
    const audio = useSelector(selectAudio);
    const isInterrupt = useSelector(selectInterrupt); // new
    const dispatch = useDispatch();
    const playerRef = useRef(); // new
    // new useEffect
    useEffect(() => {
        if (isInterrupt.value) {
          console.log('PAUSING AUDIO')
          playerRef.current.audioEl.current.pause(); // pause is a safe method, thus we do not check if audio player is playing before executing
          dispatch(clearAudioPostInterruption()); // Causes it to play twice
        }
      }, [isInterrupt]);

    return <ReactAudioPlayer ref={playerRef}
                             src={audio}
                             autoPlay
                             onEnded={() => dispatch(clearAudio())}
                             onPlay={() => dispatch(changeStatus({status: PlayerStatus.RESPONDING}))}/>;
    // return <AudioPlayer ref={playerRef}
    //                     src={audio}
    //                     autoPlay onEnded={() => {dispatch(clearAudio())}}
    //                     onPlay={() => dispatch(changeStatus({status: PlayerStatus.RESPONDING}))}/>;
};

export default SimpleRecorder;