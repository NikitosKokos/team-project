import React, { useState } from 'react';
import s from './styles.module.scss';

const Rubrics = ({ currentRubric, setCurrentRubric }) => {
   const [rubrics, setRubrics] = useState([
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

   const handleRubricClick = (rubric) => {
      setCurrentRubric(rubric);
   };
   return (
      <div className={s.rubrics}>
         <ul className={s.rubrics__list}>
            {rubrics.map(({ id, name }) => (
               <li
                  key={id}
                  onClick={() => handleRubricClick({ id, name })}
                  className={`${s.rubrics__item} ${currentRubric?.id === id ? s.active : ''}`}>
                  <div className={s.rubrics__title}>{name}</div>
               </li>
            ))}
         </ul>
      </div>
   );
};

export default Rubrics;
