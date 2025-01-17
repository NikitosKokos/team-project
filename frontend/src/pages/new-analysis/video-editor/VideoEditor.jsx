import React, { useRef, useState, useEffect } from 'react';
import s from './styles.module.scss';
import VideoTicks from './video-ticks/VideoTicks';
import VideoInfo from './video-info/VideoInfo';
import VideoStages from './video-stages/VideoStages';

const VideoEditor = ({ videoSrc, setIsStagesSaved }) => {
   const [currentStage, setCurrentStage] = useState(0);
   const [rubric, setRubric] = useState({
      // id: 2,
      // name: 'Shot Put',
      video_id: '',
      stages: [
         {
            stage_name: 'stage1',
            start_time: null,
            end_time: null,
         },
         {
            stage_name: 'stage2',
            start_time: null,
            end_time: null,
         },
         {
            stage_name: 'stage3',
            start_time: null,
            end_time: null,
         },
         {
            stage_name: 'stage4',
            start_time: null,
            end_time: null,
         },
         {
            stage_name: 'stage5',
            start_time: null,
            end_time: null,
         },
      ],
   });
   const videoRef = useRef(null);
   const trackRef = useRef(null);
   const [progress, setProgress] = useState(0);
   const [duration, setDuration] = useState(0);
   const [startFrame, setStartFrame] = useState(0); // Start position in frames
   const [endFrame, setEndFrame] = useState(0); // End position in frames
   const [isDraggingStart, setIsDraggingStart] = useState(false);
   const [isDraggingEnd, setIsDraggingEnd] = useState(false);
   const [isDraggingRange, setIsDraggingRange] = useState(false);
   const [dragStartOffset, setDragStartOffset] = useState(0);
   const [frameRate, setFrameRate] = useState(30);
   const [isScrubbing, setIsScrubbing] = useState(false);
   const minFrameSelect = 10;
   const [videoLength, setVideoLength] = useState(null);
   const [lastChange, setLastChange] = useState(null);

   const saveStage = () => {
      const newStages = rubric.stages.map((stage, index) => {
         return index === currentStage
            ? { ...stage, start_time: startFrame, end_time: endFrame }
            : stage;
      });
      const newRubric = { ...rubric, stages: newStages };

      let rubricSaved = 0;
      newRubric.stages.map((stage) => {
         if (stage.start_time !== null && stage.end_time !== null) rubricSaved++;
      });
      // console.log(rubricSaved);
      if (rubricSaved === newStages.length) {
         setIsStagesSaved(true);
      }
      setRubric(newRubric);
   };

   const handleStageChange = (index) => {
      setLastChange(null);
      const newCurrentStage = rubric.stages[index];
      if (newCurrentStage.start_time !== null && newCurrentStage.end_time !== null) {
         setStartFrame(newCurrentStage.start_time);
         setEndFrame(newCurrentStage.end_time);
      } else {
         setStartFrame(0);

         setEndFrame(videoLength);
      }

      setCurrentStage(index);

      if (videoRef.current) {
         videoRef.current.pause();
      }
   };

   useEffect(() => {
      const video = videoRef.current;

      const handleLoadedMetadata = () => {
         setDuration(video.duration); // Set the video duration
      };

      if (video) {
         video.addEventListener('loadedmetadata', handleLoadedMetadata);
      }

      return () => {
         if (video) {
            video.removeEventListener('loadedmetadata', handleLoadedMetadata);
         }
      };
   }, []);

   useEffect(() => {
      const updateProgress = () => {
         const video = videoRef.current;
         if (video && !isScrubbing) {
            const percentage = (video.currentTime / video.duration) * 100;
            setProgress(percentage);
         }
      };

      const video = videoRef.current;
      if (video) {
         video.addEventListener('timeupdate', updateProgress);
      }

      return () => {
         if (video) {
            video.removeEventListener('timeupdate', updateProgress);
         }
      };
   }, [isScrubbing]);

   const handleScrub = (e) => {
      const track = trackRef.current;
      const video = videoRef.current;

      if (track && video) {
         const rect = track.getBoundingClientRect();
         const clientX = e.clientX || e.touches[0].clientX;
         const clickX = Math.max(0, Math.min(clientX - rect.left, rect.width));
         const clickPercentage = clickX / rect.width;
         video.currentTime = clickPercentage * video.duration;
         setProgress(clickPercentage * 100);
      }
   };

   useEffect(() => {
      const video = videoRef.current;

      const handleLoadedMetadata = () => {
         setDuration(video.duration); // Set the video duration in seconds

         // Extract frame rate dynamically (approximated using total frames)
         const totalFrames = video.webkitVideoDecodedByteCount || video.duration * 30; // Use a fallback frame rate of 30
         setFrameRate(totalFrames / video.duration);

         const newVideoLength = Math.floor(video.duration * frameRate);
         setVideoLength(newVideoLength);
         setEndFrame(newVideoLength); // Set end position as total frames
      };

      if (video) {
         video.addEventListener('loadedmetadata', handleLoadedMetadata);
      }

      return () => {
         if (video) {
            video.removeEventListener('loadedmetadata', handleLoadedMetadata);
         }
      };
   }, [frameRate]);

   useEffect(() => {
      const updateProgress = () => {
         const video = videoRef.current;
         if (video) {
            const percentage = (video.currentTime / video.duration) * 100;
            setProgress(percentage);
         }
      };

      const video = videoRef.current;
      if (video) {
         video.addEventListener('timeupdate', updateProgress);
      }

      return () => {
         if (video) {
            video.removeEventListener('timeupdate', updateProgress);
         }
      };
   }, []);

   const startScrubbing = (e) => {
      setIsScrubbing(true);
      handleScrub(e);
   };

   const scrubbing = (e) => {
      if (isScrubbing) {
         handleScrub(e);
      }
   };

   const stopScrubbing = () => {
      setIsScrubbing(false);
   };

   useEffect(() => {
      const handleKeyDown = (e) => {
         const video = videoRef.current;
         if (!video) return;

         const shiftMultiplier = e.shiftKey ? 10 : 1;
         const frameTime = 1 / frameRate; // Time per frame in seconds

         switch (e.key.toLowerCase()) {
            case 'a': // Move backward
            case 'arrowleft':
               video.currentTime = Math.max(video.currentTime - frameTime * shiftMultiplier, 0);
               break;
            case 'd': // Move forward
            case 'arrowright':
               video.currentTime = Math.min(
                  video.currentTime + frameTime * shiftMultiplier,
                  duration,
               );
               break;
            case 'b': // create a breakpoint
               if (lastChange) {
                  const frameTime = 1 / frameRate; // Time per frame in seconds
                  const currentFrame = Math.round(video.currentTime * frameRate); // Current frame

                  if (lastChange === 'start') {
                     if (endFrame - minFrameSelect > currentFrame) {
                        setStartFrame(currentFrame);
                     }
                  } else if (lastChange === 'end') {
                     // setEndFrame(currentFrame);
                     if (startFrame + minFrameSelect < currentFrame) {
                        setEndFrame(currentFrame);
                     }
                  }
               }
               break;
            default:
               break;
         }
      };

      window.addEventListener('keydown', handleKeyDown);
      return () => {
         window.removeEventListener('keydown', handleKeyDown);
      };
   }, [frameRate, duration, lastChange]);

   useEffect(() => {
      if (isScrubbing) {
         document.addEventListener('mousemove', scrubbing);
         document.addEventListener('mouseup', stopScrubbing);
         document.addEventListener('touchmove', scrubbing);
         document.addEventListener('touchend', stopScrubbing);
      } else {
         document.removeEventListener('mousemove', scrubbing);
         document.removeEventListener('mouseup', stopScrubbing);
         document.removeEventListener('touchmove', scrubbing);
         document.removeEventListener('touchend', stopScrubbing);
      }

      return () => {
         document.removeEventListener('mousemove', scrubbing);
         document.removeEventListener('mouseup', stopScrubbing);
         document.removeEventListener('touchmove', scrubbing);
         document.removeEventListener('touchend', stopScrubbing);
      };
   }, [isScrubbing]);

   useEffect(() => {
      if (isDraggingStart || isDraggingEnd) {
         document.addEventListener('mousemove', handleDragging);
         document.addEventListener('mouseup', handleDragEnd);
         document.addEventListener('touchmove', handleDragging);
         document.addEventListener('touchend', handleDragEnd);
      }

      return () => {
         document.removeEventListener('mousemove', handleDragging);
         document.removeEventListener('mouseup', handleDragEnd);
         document.removeEventListener('touchmove', handleDragging);
         document.removeEventListener('touchend', handleDragEnd);
      };
   }, [isDraggingStart, isDraggingEnd]);

   const handleDragStart = (e, type) => {
      if (type === 'start') {
         setIsDraggingStart(true);
         setLastChange('start');
         console.log('set start');
      } else if (type === 'end') {
         setIsDraggingEnd(true);
         setLastChange('end');
         console.log('set end');
      }
   };

   const handleDragging = (e) => {
      const track = trackRef.current;

      if (track) {
         const rect = track.getBoundingClientRect();
         const clientX = e.clientX || e.touches?.[0]?.clientX;
         const positionX = Math.max(0, Math.min(clientX - rect.left, rect.width));
         const totalFrames = Math.floor(duration * frameRate);
         const newFrame = Math.round((positionX / rect.width) * totalFrames);

         if (isDraggingStart) {
            if (newFrame < endFrame - minFrameSelect) {
               handleScrub(e);
               setStartFrame(newFrame);
            }
         } else if (isDraggingEnd) {
            if (newFrame > startFrame + minFrameSelect) {
               handleScrub(e);
               setEndFrame(newFrame);
            }
         }
      }
   };

   const handleDragEnd = () => {
      setIsDraggingStart(false);
      setIsDraggingEnd(false);
      // console.log(`Start Frame: ${startFrame}, End Frame: ${endFrame}`);
   };

   const calculatePositionPercentage = (frame) => {
      const totalFrames = Math.floor(duration * frameRate);
      return (frame / totalFrames) * 100;
   };

   return (
      <div className={s.videoEditor}>
         <VideoStages
            rubric={rubric}
            setRubric={setRubric}
            currentStage={currentStage}
            setCurrentStage={setCurrentStage}
            saveStage={saveStage}
            handleStageChange={handleStageChange}
         />
         <div className={s.videoEditor__body}>
            <VideoInfo />
            <video ref={videoRef} src={videoSrc} controls className={s.videoEditor__video}></video>
         </div>
         <VideoTicks duration={duration} startScrubbing={startScrubbing} />
         <div ref={trackRef} className={s.videoEditor__track}>
            {/* Range Highlight */}
            <div
               className={s.videoEditor__rangeHighlight}
               style={{
                  left: `${calculatePositionPercentage(startFrame)}%`,
                  width: `${calculatePositionPercentage(endFrame - startFrame)}%`,
               }}
               // onMouseDown={(e) => handleDragStart(e, 'range')}
               // onTouchStart={(e) => handleDragStart(e, 'range')}
            ></div>
            <div>
               {/* Start Handle */}
               <div
                  className={s.videoEditor__rangeHandle}
                  style={{ left: `${calculatePositionPercentage(startFrame)}%` }}
                  onMouseDown={(e) => handleDragStart(e, 'start')}
                  onTouchStart={(e) => handleDragStart(e, 'start')}></div>

               {/* End Handle */}
               <div
                  className={s.videoEditor__rangeHandle}
                  style={{ left: `${calculatePositionPercentage(endFrame)}%` }}
                  onMouseDown={(e) => handleDragStart(e, 'end')}
                  onTouchStart={(e) => handleDragStart(e, 'end')}></div>
            </div>

            {/* Progress */}
            <div className={s.videoEditor__progress} style={{ left: `${progress}%` }}>
               <div className={s.videoEditor__progressBody}>
                  <div className={s.videoEditor__progressTriangle}></div>
                  <div className={s.videoEditor__progressLine}></div>
               </div>
            </div>
         </div>
      </div>
   );
};

export default VideoEditor;
