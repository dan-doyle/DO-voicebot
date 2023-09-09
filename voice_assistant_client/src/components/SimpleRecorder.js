import React from "react";
import PropTypes from "prop-types";
import {connect} from "react-redux";
import {Button, Icon} from "semantic-ui-react";

import {changeStatus, hotwordResponse, foundHotword, userInterruptDetected} from "../reducers/media"; // new userInterrupt
import {queryInterruption, submitHotwordRecording, submitRecording} from "../reducers/socket";
import {setupAudioRecorder, setupHark} from "../util";
import {PlayerStatus} from "../reducers/const";

import "react-loader-spinner/dist/loader/css/react-spinner-loader.css";
import "./SimpleRecorder.css";

class SimpleRecorder extends React.Component {
    constructor(props) {
        super(props);
        this.overlapSpeech = false // NEW // no need to include in state, unlike recordings which needs a re-render when changed 
        this.state = {recorder: null, hark: null, backgroundRecorder: null, backgroundHark: null, interruptRecorder: null, interruptHark: null, recordings: null}; // NEW // interruptHark implemented
    }

    componentDidMount() {
        this.listenForHotword()
    }

    componentDidUpdate(prevProps) {
        if (prevProps.status != this.props.status) {
            console.log('Changed status to ', this.props.status)
            if (this.props.status === PlayerStatus.IDLE) {
                this.listenForHotword();
            }
            // --------START OF NEW CODE SECTION--------
            // when change to responding we set up an interruption recorder
            if (this.props.status === PlayerStatus.RESPONDING) {
                this.listenForInterruption();
                this.setState({recordings: null}) // clear out recordings from any previous turn
            }
            
            if (prevProps.status == PlayerStatus.RESPONDING && !this.overlapSpeech) {
                // Only if no interruption related recording is taking place, we clear hark and recorder if there is one
                if (this.state.interruptRecorder) {
                    this.state.interruptRecorder.destroy();
                    this.setState({interruptRecorder: null});
                }
                this.state.interruptHark.stop();
                this.setState({interruptHark: null});
            }
            // --------END OF NEW CODE SECTION--------
        }

        if (prevProps.receivedHotwordRes != this.props.receivedHotwordRes || 
                prevProps.detectedHotword != this.props.detectedHotword) {
            if (this.props.detectedHotword) {
                this.onRecordClick()
                this.props.dispatch(foundHotword(false));
            } else {
                this.listenForHotword()
            }

            this.props.dispatch(hotwordResponse(false));
        }
        // --------START OF NEW CODE SECTION--------
        // if we register either a new interruption or a new recording, check to see if we need to submit recording !
        console.log('IN onComponentDidUpdate, checking isInterrupt: ', this.props.isInterrupt.value, ' AND ', this.state.recordings)
        if (this.props.isInterrupt.value && this.state.recordings) {
            console.log('SEARCHING FOR RECORDING WITH UUID: ', this.props.isInterrupt.id);
            // send the recording corresponding to id in isInterrupt
            const targetRecording = this.state.recordings.find(
                recording => recording.id === this.props.isInterrupt.id
            );
            if (targetRecording) {  // if the recording with the matching id is found
                this.props.dispatch(submitRecording(targetRecording)); // start process of creating next turn of conversation
                console.log('RECORDING SUBMITTED WITH UUID: ', this.props.isInterrupt.id);
                // barge-in officially over
                this.props.dispatch(userInterruptDetected({value: false, id: null}));
            } else {
                console.warn('Recording with matching ID not found.');
            }
            // recordings will be cleared when the chatbot status returns to RECORDING for the next turn
        }
        // --------END OF NEW CODE SECTION--------
    }

    listenForHotword() {
        if (this.props.status === PlayerStatus.IDLE) {
            // console.log('Listening in the background')
            if (this.state.backgroundRecorder) {
                this.state.backgroundRecorder.destroy();
                this.setState({backgroundRecorder: null});
            }

            setupAudioRecorder().then(backgroundRecorder => {
                this.setState({backgroundRecorder});
                backgroundRecorder.startRecording();
            });

            setupHark().then(backgroundHark => {
                this.setState({backgroundHark});
                backgroundHark.on("speaking", () => {
                    // console.log("Hotword speaking");
                });

                backgroundHark.on("stopped_speaking", () => {
                    // console.log("Hotword stopped talking");
                    this.sendHotwordRecording()
                });
            });
        }
    }

    sendHotwordRecording() {
        if (this.props.status === PlayerStatus.IDLE && this.state.backgroundRecorder.state === "recording") {
            this.state.backgroundRecorder.stopRecording(() => {
                this.state.backgroundRecorder.getDataURL((audioDataURL) => {
                    submitHotwordRecording({
                        audio: {
                            type: this.state.backgroundRecorder.getBlob().type || "audio/wav",
                            sampleRate: this.state.backgroundRecorder.sampleRate,
                            bufferSize: this.state.backgroundRecorder.bufferSize,
                            data: audioDataURL.split(",").pop()
                        }
                    })
                });
            });
        }

    }

    onRecordClick() {
        if (this.props.status === PlayerStatus.IDLE) {
            this.props.dispatch(changeStatus({status: PlayerStatus.LISTENING}));
            if (this.state.recorder) {
                this.state.recorder.destroy();
                this.setState({recorder: null});
            }

            setupAudioRecorder().then(recorder => {
                this.setState({recorder});
                recorder.clearRecordedData();
                recorder.startRecording();
            });

            if (this.state.hark) {
                this.state.hark.stop();
                this.setState({hark: null});
            }

            setupHark().then(hark => {
                this.setState({hark});
                hark.on("speaking", () => {
                    // console.log("speaking");
                });

                hark.on("stopped_speaking", () => {
                    // console.log("stopped talking");
                    this.onStopClick();
                });
            });
        }
    }

    onStopClick() {
        if (this.props.status === PlayerStatus.LISTENING && this.state.recorder) {
            this.state.recorder.stopRecording(() => {
                this.state.recorder.getDataURL((audioDataURL) => {
                    this.props.dispatch(submitRecording({
                        audio: {
                            type: this.state.recorder.getBlob().type || "audio/wav",
                            sampleRate: this.state.recorder.sampleRate,
                            bufferSize: this.state.recorder.bufferSize,
                            data: audioDataURL.split(",").pop(),
                            overlappedSpeech: false,
                        }
                    }));
                });
            });
        }

        if (this.state.hark) {
            this.state.hark.stop();
        }

        this.listenForHotword()
    }

    // --------NEW FUNCTION: listenForInterruption--------
    listenForInterruption() {
        if (!this.state.interruptRecorder) {
            setupAudioRecorder(true, queryInterruption).then(interruptRecorder => {
                this.setState({interruptRecorder});
                // startRecording() is called at 'start speaking'
            });
        }
        if (!this.state.interruptHark) {            
            //Hark works in parallel with the recorders to and informs them of when to start / stop
            setupHark().then(interruptHark => {
                this.setState({interruptHark});
                interruptHark.on("speaking", () => {
                    this.state.interruptRecorder.startRecording();
                    console.log("Interruption speech detected by hark, starting recording");
                    this.overlapSpeech = true;
                });

                interruptHark.on("stopped_speaking", () => {
                    // Logic: add audio to recordings and if RESPONDING has finished by the time this is done, clean up
                    this.state.interruptRecorder.stopRecording(() => {
                        this.state.interruptRecorder.getDataURL((audioDataURL) => {
                            // we save the audio to recordings which will allow it to be submitted if deemed an interruption
                            this.setState(prevState => {
                                const newRecording = {
                                    id: this.state.interruptRecorder.getInterruptId(),
                                    audio: {
                                        type: prevState.interruptRecorder.getBlob().type || "audio/wav",
                                        sampleRate: prevState.interruptRecorder.getSampleRate(), // these are now getter method calls rather than attributes of the direct recorder
                                        bufferSize: prevState.interruptRecorder.getBufferSize(),
                                        data: audioDataURL.split(",").pop(),
                                        overlappedSpeech: true,
                                    }
                                };
                                const updatedRecordings = prevState.recordings ? [...prevState.recordings, newRecording] : [newRecording];
                                return { recordings: updatedRecordings };
                            });

                            // All below code within callback placed here to allow the above async code to finish, it must be ran after
                            this.overlapSpeech = false;

                            this.state.interruptRecorder.destroy(); // these two lines can be redundant if we make overlapSpeech part of state
                            this.setState({interruptRecorder: null});

                        
                            if (this.props.status != PlayerStatus.RESPONDING) {
                                // if we enter here, this means that the other way of cleaning up the hark has been bypassed, which is when we shift away from RESPONDING and there is no overlapped speech
                                this.state.interruptHark.stop();
                                this.setState({interruptHark: null});
                            } else {
                                setupAudioRecorder(true, queryInterruption).then(interruptRecorder => {
                                    this.setState({interruptRecorder});
                                    // the corresponding startRecording() call is located at the 'start speaking' event
                                });
                            }
                        });
                    });
                });
            });
        }
    }

    recordColor(status) {
        switch (status) {
            case PlayerStatus.LISTENING:
                return "green";
            default:
                return null;
        }
    }

    render() {
        return <Button.Group className="toolbar-margin">
            <Button icon onClick={this.onRecordClick.bind(this)} disabled={this.props.status !== PlayerStatus.IDLE}>
                <Icon name="record" color={this.recordColor(this.props.status)}/>
            </Button>
            <Button icon onClick={this.onStopClick.bind(this)} disabled={this.props.status !== PlayerStatus.LISTENING}>
                <Icon name="stop" color="red"/>
            </Button>
        </Button.Group>;
    }
}

SimpleRecorder.propTypes = {
    status: PropTypes.string,
    dispatch: PropTypes.func
};

const mapStateToProps = state => ({status: state.media.status, receivedHotwordRes: state.media.receivedHotwordRes, detectedHotword: state.media.detectedHotword, isInterrupt: state.media.isInterrupt, queryInterruption }); // NEW, added isInterrupt + queryInterruption

export default connect(mapStateToProps)(SimpleRecorder);