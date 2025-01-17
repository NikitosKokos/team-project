import React, { useState } from 'react';
import s from './styles.module.scss';

const VideoStages = ({
   rubric,
   setRubric,
   currentStage,
   setCurrentStage,
   saveStage,
   handleStageChange,
}) => {
   return (
      <div className={s.videoStages}>
         <div className={s.videoStages__body}>
            <ul className={s.videoStages__list}>
               {rubric.stages.map((stage, index) => (
                  <div
                     key={index}
                     onClick={() => handleStageChange(index)}
                     className={`${s.videoStages__item} ${currentStage === index ? s.active : ''} ${
                        stage.start_time !== null && stage.end_time !== null ? s.completed : ''
                     }`}>
                     {index + 1}
                  </div>
               ))}
            </ul>
            <button className={s.videoStages__save} onClick={saveStage}>
               Save Stage
            </button>
         </div>
      </div>
   );
};

export default VideoStages;
