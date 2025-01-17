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
   const [fileName, setFileName] = useState(null);
   const [isUserChosen, setIsUserChosen] = useState(false);
   const [isStagesSaved, setIsStagesSaved] = useState(false);
   const [stages, setStages] = useState([
      { id: 0, name: 'Start' },
      { id: 1, name: 'Sprint' },
      { id: 2, name: 'Shot Put' },
      { id: 3, name: 'Height Jump' },
      { id: 4, name: 'Hurdles (official spacing)' },
      { id: 5, name: 'Long Jump' },
      { id: 6, name: 'Discus Throw' },
      { id: 7, name: 'Javelin Throw' },
      { id: 8, name: 'Relay Race' },
   ]);

   const handleVideoUpload = (file) => {
      const fileURL = URL.createObjectURL(file);
      setVideoSrc(fileURL);
   };

   const handleSubmit = () => {
      if (isStagesSaved) {
         // axios logic
      } else {
         if (currentRubric && fileName && isUserChosen) {
            console.log(fileName);

            setShowVideoEditor(true);
         }
      }
   };

   return (
      <div className={s.newAnalysis}>
         <div className={s.newAnalysis__main}>
            <div className={s.newAnalysis__left}>
               <div className={s.newAnalysis__title}>Create a new analysis</div>
               <ChooseStudent setIsUserChosen={setIsUserChosen} />
               <UploadVideo onUpload={handleVideoUpload} setFileName={setFileName} />
               <button className={s.newAnalysis__submit} onClick={handleSubmit}>
                  {!isStagesSaved ? 'Submit' : 'Analyze'}
               </button>
            </div>
            {showVideoEditor ? (
               <VideoEditor videoSrc={videoSrc} setIsStagesSaved={setIsStagesSaved} />
            ) : (
               <Rubrics currentRubric={currentRubric} setCurrentRubric={setCurrentRubric} />
            )}
         </div>
      </div>
   );
};

export default NewAnalysis;
