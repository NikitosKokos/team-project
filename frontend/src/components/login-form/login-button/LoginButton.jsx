import React from 'react';
import s from './styles.module.scss';

const LoginButton = () => {
   return (
      <button className={s.btn}>
         <div className={s.btn__icon}>
            <svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
               <path d="M1.24512 1.19453H9.55879V9.50743H1.24512V1.19453Z" fill="#565356" />
               <path d="M10.4258 1.19453H18.7387V9.50743H10.4258V1.19453Z" fill="#565356" />
               <path d="M1.24512 10.3754H9.55879V18.6879H1.24512V10.3754Z" fill="#565356" />
               <path d="M10.4258 10.3754H18.7387V18.6879H10.4258V10.3754Z" fill="#565356" />
            </svg>
         </div>
         <div className={s.btn__main}>
            <div className={s.btn__text}>Continue with Microsoft</div>
         </div>
      </button>
   );
};

export default LoginButton;
