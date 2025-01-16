import React, { useRef, useState, useEffect } from 'react';
import s from './styles.module.scss';
import VideoTicks from './video-ticks/VideoTicks';

const VideoEditor = ({ videoSrc }) => {
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

         setEndFrame(Math.floor(video.duration * frameRate)); // Set end position as total frames
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

   const handleDragStart = (e, type) => {
      if (type === 'start') {
         setIsDraggingStart(true);
      } else if (type === 'end') {
         setIsDraggingEnd(true);
      } else if (type === 'range') {
         setIsDraggingRange(true);
         const track = trackRef.current;
         const rect = track.getBoundingClientRect();
         const clientX = e.clientX || e.touches?.[0]?.clientX;
         setDragStartOffset(clientX - rect.left);
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
            if (newFrame < endFrame - 100) {
               setStartFrame(newFrame);
            }
         } else if (isDraggingEnd) {
            if (newFrame > startFrame + 100) {
               setEndFrame(newFrame);
            }
         } else if (isDraggingRange) {
            const dragOffsetFrames = Math.round((dragStartOffset / rect.width) * totalFrames);
            const rangeWidth = endFrame - startFrame;
            const newStartFrame = newFrame - dragOffsetFrames;

            // Ensure the range stays within bounds
            if (newStartFrame >= 0 && newStartFrame + rangeWidth <= totalFrames) {
               setStartFrame(newStartFrame);
               setEndFrame(newStartFrame + rangeWidth);
            }
         }
      }
   };

   const handleDragEnd = () => {
      setIsDraggingStart(false);
      setIsDraggingEnd(false);
      setIsDraggingRange(false);
      console.log(`Start Frame: ${startFrame}, End Frame: ${endFrame}`);
   };

   const calculatePositionPercentage = (frame) => {
      const totalFrames = Math.floor(duration * frameRate);
      return (frame / totalFrames) * 100;
   };

   return (
      <div className={s.videoEditor}>
         <div className={s.videoEditor__body}>
            <video ref={videoRef} src={videoSrc} controls className={s.videoEditor__video}></video>
         </div>
         <VideoTicks duration={duration} />
         <div
            ref={trackRef}
            className={s.videoEditor__track}
            onMouseMove={handleDragging}
            onMouseUp={handleDragEnd}
            onTouchMove={handleDragging}
            onTouchEnd={handleDragEnd}
            onMouseDown={startScrubbing}
            onTouchStart={startScrubbing}>
            {/* Range Highlight */}
            <div
               className={s.videoEditor__rangeHighlight}
               style={{
                  left: `${calculatePositionPercentage(startFrame)}%`,
                  width: `${calculatePositionPercentage(endFrame - startFrame)}%`,
               }}
               onMouseDown={(e) => handleDragStart(e, 'range')}
               onTouchStart={(e) => handleDragStart(e, 'range')}></div>
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
            <div className={s.videoEditor__progress} style={{ left: `${progress}%` }}></div>
         </div>
      </div>
   );
};

export default VideoEditor;
