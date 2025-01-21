import React from 'react';
import s from './styles.module.scss';

const Stages = ({ stages }) => {
   return (
      <div className={s.stages}>
         <ul className={s.stages__list}>
            {stages.map((el, index) => (
               <li key={index} className={`${s.stages__item} ${s.rede}`}>
                  {el}
               </li>
            ))}
         </ul>
      </div>
   );
};

export default Stages;
