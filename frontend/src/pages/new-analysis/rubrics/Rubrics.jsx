import React, { useState } from 'react';
import s from './styles.module.scss';

const Rubrics = () => {
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
   return (
      <div className={s.rubrics}>
         <ul className={s.rubrics__list}>
            {rubrics.map(({ id, name }) => (
               <li key={id} className={s.rubrics__item}>
                  <div className={s.rubrics__title}>{name}</div>
               </li>
            ))}
         </ul>
      </div>
   );
};

export default Rubrics;
