import React from 'react';
import s from './styles.module.scss';
import UploadVideo from './upload-video/UploadVideo';
import ChooseStudent from './choose-student/ChooseStudent';
import Rubrics from './rubrics/Rubrics';

const NewAnalysis = () => {
   return (
      <div className={s.newAnalysis}>
         <div className={s.newAnalysis__main}>
            <div className={s.newAnalysis__left}>
               <div className={s.newAnalysis__title}>Create a new analysis</div>
               <ChooseStudent />
               <UploadVideo />
            </div>
            <Rubrics />
         </div>
      </div>
   );
};

export default NewAnalysis;
