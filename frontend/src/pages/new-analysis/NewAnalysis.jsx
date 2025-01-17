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
   const [isLoading, setIsLoading] = useState(false);
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

   const handleVideoUpload = (file) => {
      const fileURL = URL.createObjectURL(file);
      setVideoSrc(fileURL);
   };

   const handleSubmit = () => {
      if (currentRubric && fileName && isUserChosen) {
         console.log(fileName);

         setShowVideoEditor(true);
      }
   };

   const handleAnalyze = async () => {
      if (isStagesSaved) {
         const newRubric = { ...rubric, video_id: fileName };
         setRubric(newRubric);
         setIsLoading(true);
         console.log(newRubric);

         // axios logic
         async function getSasForFile(filename) {
            const baseUrl = 'https://evaluation-scripts.azurewebsites.net/api/get_sas';
            const url = `${baseUrl}?filename=${encodeURIComponent(filename)}`;

            const response = await fetch(url, { method: 'GET' });
            if (!response.ok) {
               throw new Error(`SAS error: HTTP ${response.status}`);
            }
            const data = await response.json();
            return data.sas_url; // e.g. "https://account.blob.core.windows.net/container/filename.mp4?someSAS"

            async function uploadFile(file) {
               const sasUrl = await getSasForFile(file.name);

               // PUT the file
               const res = await fetch(sasUrl, {
                  method: 'PUT',
                  headers: {
                     'x-ms-blob-type': 'BlockBlob',
                     'Content-Type': file.type,
                  },
                  body: file,
               });
               if (!res.ok) {
                  throw new Error(`Upload failed: ${res.status}`);
               }
               console.log('Upload success to', sasUrl);
               return sasUrl; // store for future reference
            }
         }

         await getSasForFile(videoSrc);
      }
   };

   return (
      <div className={s.newAnalysis}>
         <div className={s.newAnalysis__main}>
            <div className={s.newAnalysis__left}>
               <div className={s.newAnalysis__title}>Create a new analysis</div>
               {!showVideoEditor ? (
                  <>
                     <ChooseStudent setIsUserChosen={setIsUserChosen} />
                     <UploadVideo onUpload={handleVideoUpload} setFileName={setFileName} />
                     <button className={s.newAnalysis__submit} onClick={handleSubmit}>
                        Submit
                     </button>
                  </>
               ) : (
                  <button
                     className={`${s.newAnalysis__submit} ${isLoading ? s.disabled : ''}`}
                     onClick={handleAnalyze}>
                     {!isLoading ? 'Analyze' : 'Loading...'}
                  </button>
               )}
            </div>
            {showVideoEditor ? (
               <VideoEditor
                  videoSrc={videoSrc}
                  setIsStagesSaved={setIsStagesSaved}
                  rubric={rubric}
                  setRubric={setRubric}
               />
            ) : (
               <Rubrics currentRubric={currentRubric} setCurrentRubric={setCurrentRubric} />
            )}
         </div>
      </div>
   );
};

export default NewAnalysis;
