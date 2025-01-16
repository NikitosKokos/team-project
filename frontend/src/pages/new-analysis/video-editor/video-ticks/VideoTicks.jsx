import React from 'react';
import s from './styles.module.scss';

const VideoTicks = ({ duration, startScrubbing }) => {
   const formatTime = (time) => {
      const minutes = Math.floor(time / 60);
      const seconds = Math.floor(time % 60)
         .toString()
         .padStart(2, '0');
      return `${minutes}:${seconds}`;
   };

   const renderTicks = () => {
      const numTicks = 14; // Fixed number of ticks
      const interval = duration / numTicks; // Interval between ticks
      const tickElements = [];

      for (let i = 0; i <= numTicks; i++) {
         const time = i * interval;
         const percentage = (time / duration) * 100;
         tickElements.push(
            <div key={i} className={s.videoTicks__tick} style={{ left: `${percentage}%` }}>
               <span className={s.videoTicks__timestamp}>{formatTime(time)}</span>
            </div>,
         );
      }

      return tickElements;
   };
   return (
      <div className={s.videoTicks} onMouseDown={startScrubbing} onTouchStart={startScrubbing}>
         <div className={s.videoTicks__body}>{renderTicks()}</div>
      </div>
   );
};

export default VideoTicks;
