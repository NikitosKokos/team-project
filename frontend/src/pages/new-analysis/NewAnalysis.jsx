import React, { useState } from 'react';
import s from './styles.module.scss';
import UploadVideo from './upload-video/UploadVideo';
import ChooseStudent from './choose-student/ChooseStudent';
import Rubrics from './rubrics/Rubrics';
import VideoEditor from './video-editor/VideoEditor';

const NewAnalysis = () => {
   const [showVideoEditor, setShowVideoEditor] = useState(false);
   const [videoSrc, setVideoSrc] = useState('');
   const [currentRubric, setCurrentRubric] = useState(null);

   const handleVideoUpload = (file) => {
      const fileURL = URL.createObjectURL(file);
      setVideoSrc(fileURL);
   };

   const handleSubmit = () => {
      console.log(currentRubric);
   };

   return (
      <div className={s.newAnalysis}>
         <div className={s.newAnalysis__main}>
            <div className={s.newAnalysis__left}>
               <div className={s.newAnalysis__title}>Create a new analysis</div>
               <ChooseStudent />
               <UploadVideo onUpload={handleVideoUpload} />
               <button className={s.newAnalysis__submit} onClick={handleSubmit}>
                  Submit
               </button>
            </div>
            {showVideoEditor ? (
               <VideoEditor videoSrc={videoSrc} />
            ) : (
               <Rubrics currentRubric={currentRubric} setCurrentRubric={setCurrentRubric} />
            )}
         </div>
      </div>
   );
};

export default NewAnalysis;
