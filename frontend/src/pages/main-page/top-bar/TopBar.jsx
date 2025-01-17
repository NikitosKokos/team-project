import React from 'react';
import { NavLink } from 'react-router';
import s from './styles.module.scss';

const TopBar = () => {
   return (
      <div className={s.topBar}>
         <ul className={s.topBar__list}>
            <li className={s.topBar__item}>
               <NavLink
                  to="/"
                  className={({ isActive }) =>
                     isActive ? `${s.topBar__link} ${s.topBar__link_active}` : s.topBar__link
                  }>
                  New analysis
               </NavLink>
            </li>
            <li className={s.topBar__item}>
               <NavLink
                  to="/history"
                  className={({ isActive }) =>
                     isActive ? `${s.topBar__link} ${s.topBar__link_active}` : s.topBar__link
                  }>
                  History
               </NavLink>
            </li>
         </ul>
      </div>
   );
};

export default TopBar;
