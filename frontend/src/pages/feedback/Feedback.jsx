import React, { useEffect, useState } from 'react';
import s from './styles.module.scss';
import { useParams } from 'react-router';
import Stages from './stages/Stages';

const Feedback = ({ rubrics }) => {
   const { userId, rubricId } = useParams();
   const [title, setTitle] = useState('Not Found');
   const [isLoading, setIsLoading] = useState(true);
   const [scores, setScores] = useState([]);
   const [totalPoint, setTotalPoint] = useState(null);

   useEffect(() => {
      if (rubrics.length > rubricId && rubricId >= 0) {
         setTitle(rubrics[rubricId].name);
      }
      if (isLoading) {
         setTimeout(() => {
            setIsLoading(false);
            const newScores = [0.5, 0, 1, 1, 1];
            setScores(newScores);
            setTotalPoint(newScores.reduce((partialSum, a) => partialSum + a, 0));
         }, 2000);
      }
   }, []);

   return (
      <div className={s.feedback}>
         <div className={s.feedback__top}>
            <h1 className={s.feedback__title}>Rubric: {title}</h1>
            <div className={s.feedback__score}>{totalPoint}/5</div>
         </div>
         <Stages stages={rubrics[rubricId].rubrics} scores={scores} />
         <p>User ID: {userId}</p>
         <p>Rubric Name: {rubricId}</p>
      </div>
   );
};

export default Feedback;
