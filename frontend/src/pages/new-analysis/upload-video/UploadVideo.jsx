import React from 'react';
import s from './styles.module.scss';

const UploadVideo = () => {
   const handleUpload = (event) => {
      const file = event.target.files[0];
      if (file) {
         console.log('File selected:', file.name);
         // Handle the uploaded video file (e.g., send to server)
      }
   };

   return (
      <div className={s.uploadVideo}>
         <div className={s.uploadVideo__body}>
            <label htmlFor="videoUpload" className={s.uploadButton}>
               <div className={s.uploadIcon}>
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512">
                     <path d="M246.6 9.4c-12.5-12.5-32.8-12.5-45.3 0l-128 128c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0L192 109.3 192 320c0 17.7 14.3 32 32 32s32-14.3 32-32l0-210.7 73.4 73.4c12.5 12.5 32.8 12.5 45.3 0s12.5-32.8 0-45.3l-128-128zM64 352c0-17.7-14.3-32-32-32s-32 14.3-32 32l0 64c0 53 43 96 96 96l256 0c53 0 96-43 96-96l0-64c0-17.7-14.3-32-32-32s-32 14.3-32 32l0 64c0 17.7-14.3 32-32 32L96 448c-17.7 0-32-14.3-32-32l0-64z" />
                  </svg>
               </div>
               Upload Video
            </label>
            <input
               type="file"
               id="videoUpload"
               accept="video/*"
               className={s.uploadInput}
               onChange={handleUpload}
            />
         </div>
      </div>
   );
};

export default UploadVideo;
