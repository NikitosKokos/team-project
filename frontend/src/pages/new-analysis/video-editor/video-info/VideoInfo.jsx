import React, { useState } from 'react';
import s from './styles.module.scss';

const VideoInfo = () => {
   const [isPopupOpen, setIsPopupOpen] = useState(false);

   const handleInfoClick = () => {
      setIsPopupOpen(true);
   };
   const handleCloseBtn = () => {
      setIsPopupOpen(false);
   };
   return (
      <div className={s.videoInfo}>
         <button className={s.videoInfo__icon} onClick={handleInfoClick}>
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
               <path
                  d="M12 16V12M12 8H12.01M22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12Z"
                  stroke="black"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
               />
            </svg>
         </button>
         <div className={`${s.videoInfo__content} ${isPopupOpen ? s.active : ''}`}>
            <div className={s.videoInfo__body}>
               <div className={s.videoInfo__title}>Tips for editing</div>
               <div className={s.videoInfo__list}>
                  <div className={s.videoInfo__item}>
                     <div className={s.videoInfo__keys}>
                        <div className={s.videoInfo__key}>
                           <svg
                              width="24"
                              height="24"
                              viewBox="0 0 24 24"
                              fill="none"
                              xmlns="http://www.w3.org/2000/svg">
                              <path
                                 d="M12 19L5 12M5 12L12 5M5 12H19"
                                 stroke="black"
                                 strokeWidth="2"
                                 strokeLinecap="round"
                                 strokeLinejoin="round"
                              />
                           </svg>
                        </div>
                        <div className={s.videoInfo__key}>
                           <svg
                              width="24"
                              height="24"
                              viewBox="0 0 24 24"
                              fill="none"
                              xmlns="http://www.w3.org/2000/svg">
                              <path
                                 d="M5 12H19M19 12L12 5M19 12L12 19"
                                 stroke="black"
                                 strokeWidth="2"
                                 strokeLinecap="round"
                                 strokeLinejoin="round"
                              />
                           </svg>
                        </div>
                        <div className={s.videoInfo__key}>
                           <span>A</span>
                        </div>
                        <div className={s.videoInfo__key}>
                           <span>D</span>
                        </div>
                     </div>
                     <div className={s.videoInfo__text}>Move one frame forward/backward</div>
                  </div>
                  <div className={s.videoInfo__item}>
                     <div className={s.videoInfo__key}>
                        <svg
                           width="24"
                           height="24"
                           viewBox="0 0 24 24"
                           fill="none"
                           xmlns="http://www.w3.org/2000/svg">
                           <path
                              d="M9 18V12H5L12 5L19 12H15V18H9Z"
                              stroke="black"
                              strokeWidth="2"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                           />
                        </svg>
                     </div>
                     <div className={s.videoInfo__text}>Hold Shift to move 10 frames</div>
                  </div>
                  <div className={s.videoInfo__item}>
                     <div className={s.videoInfo__key}>
                        <span>B</span>
                     </div>
                     <div className={s.videoInfo__text}>Make a new breakpoint</div>
                  </div>
               </div>
               <div className={s.videoInfo__close} onClick={handleCloseBtn}>
                  <span>Click here to close</span>
                  <svg
                     width="24"
                     height="24"
                     viewBox="0 0 24 24"
                     fill="none"
                     xmlns="http://www.w3.org/2000/svg">
                     <path
                        d="M15 9L9 15M9 9L15 15M22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12Z"
                        stroke="black"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                     />
                  </svg>
               </div>
            </div>
         </div>
      </div>
   );
};

export default VideoInfo;
