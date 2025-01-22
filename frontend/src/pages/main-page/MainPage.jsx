import React, { useState } from 'react';
import { Routes, Route } from 'react-router';
import TopBar from './top-bar/TopBar.jsx';
import NewAnalysis from '../new-analysis/NewAnalysis.jsx';
import Overview from '../overview/Overview.jsx';
import History from '../history/History.jsx';
import Feedback from '../feedback/Feedback.jsx';
import s from './styles.module.scss';
import { Toaster } from 'sonner';

const MainPage = () => {
   const [rubrics, setRubrics] = useState([
      {
         id: 0,
         name: 'Start',
         rubrics: [
            'Pelvis slightly higher than shoulders at "set"',
            'Head aligned with torso, looking at start line',
            'Both legs push off powerfully, causing imbalance',
            'Body positioned like "a spear" during push-off',
            'Gaze directed slightly forward toward ground',
         ],
      },
      {
         id: 1,
         name: 'Sprint',
         rubrics: [
            'Runs on the balls of the feet',
            'Knees are high',
            'Active clawing motion of legs',
            'Arms at 90° actively moving',
            'Center of mass leans forward',
         ],
      },
      {
         id: 2,
         name: 'Shot Put',
         rubrics: [
            'Glide phase initiated with bent low leg, back to throw',
            'Trailing leg pulled under pelvis after hop',
            'Bracing leg planted; push leg bent after hop',
            'Push leg extends, hip-torso-arm sequence begins',
            'Shot remains at neck until 45° release',
         ],
      },
      {
         id: 3,
         name: 'Height Jump',
         rubrics: [
            'Accelerating approach with upright posture',
            'Lean into curve',
            'Full knee lift during takeoff, arm lifted high',
            'Clears bar with an arched back',
            'Lands on mat in "L" shape, perpendicular to bar',
         ],
      },
      {
         id: 4,
         name: 'Hurdles (official spacing)',
         rubrics: [
            '8 steps in approach',
            'First hurdle cleared',
            'Fully extended lead leg passes hurdle',
            'Torso and opposite arm align with lead leg',
            'Large stride for second contact post-hurdle',
         ],
      },
      {
         id: 5,
         name: 'Long Jump',
         rubrics: [
            'Accelerating approach without slowing before takeoff',
            'Foot lands on white plank; no gaze at board',
            'Takeoff foot flat, center of mass above it',
            'Knight stance maintained during first half',
            'Landing with sliding technique',
         ],
      },
      {
         id: 6,
         name: 'Discus Throw',
         rubrics: [
            'Throwing arm swings back after initial motion',
            'Pivot initiated from ball of foot',
            'Pivot performed flat toward circle center',
            'Throw starts low-to-high with hip engagement',
            'Discus released via index finger',
         ],
      },
      {
         id: 7,
         name: 'Javelin Throw',
         rubrics: [
            'Javelin brought backward during last 5 steps',
            'Pelvis rotated, javelin fully retracted',
            'Impulse step executed',
            'Blocking step executed',
            'Throw initiated through hip-torso involvement',
         ],
      },
      {
         id: 8,
         name: 'Relay Race',
         rubrics: [
            'Receiver starts after runner crosses mark without looking back',
            'Receiver reaches maximum speed',
            'Baton exchange occurs at speed after agreed signal',
            'Baton switched hands, runner stays in lane',
            'Exchange occurs within the zone',
         ],
      },
   ]);

   return (
      <div className="wrapper">
         <Toaster richColors position="bottom-left" className="toaster" />
         <div className={s.mainPage}>
            <div className={`${s.mainPage__container} _container`}>
               <TopBar />
               <Routes>
                  <Route path="/" element={<NewAnalysis />} />
                  <Route path="/history" element={<History />} />
                  <Route
                     path="/feedback/:userId/:rubricId"
                     element={<Feedback rubrics={rubrics} />}
                  />
               </Routes>
            </div>
         </div>
      </div>
   );
};

export default MainPage;
