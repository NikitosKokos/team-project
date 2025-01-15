import React from 'react';
import s from './styles.module.scss';
import LoginButton from './login-button/LoginButton';

const LoginForm = ({ isAnimEnded }) => {
   return (
      <div className={`${s.loginFrom} ${isAnimEnded ? s.active : ''}`.trim()}>
         <div className={s.loginFrom__main}>
            <div className={s.loginFrom__container}>
               <div className={s.loginFrom__body}>
                  <div className={s.loginFrom__title}>Login</div>
                  <LoginButton />
               </div>
            </div>
         </div>
      </div>
   );
};

export default LoginForm;
