import React from 'react';
import s from './styles.module.scss';

const Stages = ({ stages, scores }) => {
   return (
      <div className={s.stages}>
         <ul className={s.stages__list}>
            {stages.map((el, index) => (
               <li
                  key={index}
                  className={`${s.stages__item} ${`${
                     scores[index] === 0 ? s.red : scores[index] === 0.5 ? s.purple : ''
                  }`}`}>
                  {el}
               </li>
            ))}
         </ul>
      </div>
   );
};

export default Stages;
