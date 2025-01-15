import React from 'react';
import { Routes, Route } from 'react-router';
import TopBar from './top-bar/TopBar.jsx';
import NewAnalysis from '../new-analysis/NewAnalysis.jsx';
import Overview from '../overview/Overview.jsx';
import History from '../history/History.jsx';
import s from './styles.module.scss';

const MainPage = () => {
   return (
      <div className="wrapper">
         <div className={s.mainPage}>
            <div className="_container">
               <TopBar />
               <Routes>
                  <Route path="/" element={<NewAnalysis />} />
                  <Route path="/history" element={<History />} />
               </Routes>
            </div>
         </div>
      </div>
   );
};

export default MainPage;
