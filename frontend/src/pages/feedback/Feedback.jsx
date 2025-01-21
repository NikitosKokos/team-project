import React, { useEffect, useState } from 'react';
import s from './styles.module.scss';
import { useParams } from 'react-router';
import Stages from './stages/Stages';

const Feedback = ({ rubrics }) => {
   const { userId, rubricId } = useParams();
   const [title, setTitle] = useState('Not Found');

   useEffect(() => {
      console.log(1);
      if (rubrics.length > rubricId && rubricId >= 0) {
         setTitle(rubrics[rubricId].name);
      }
   }, []);

   return (
      <div className={s.feedback}>
         <div className={s.feedback__top}>
            <h1 className={s.feedback__title}>Rubric: {title}</h1>
            <div className={s.feedback__score}>4.5/5</div>
         </div>
         <Stages stages={rubrics[rubricId].rubrics} />
         <p>User ID: {userId}</p>
         <p>Rubric Name: {rubricId}</p>
      </div>
   );
};

export default Feedback;
