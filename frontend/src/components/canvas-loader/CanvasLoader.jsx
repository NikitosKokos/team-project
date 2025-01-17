import { Html, useProgress } from '@react-three/drei';
import React, { useEffect } from 'react';
import s from './styles.module.scss';

const CanvasLoader = () => {
   const { progress } = useProgress();

   useEffect(() => {
      console.log(progress);
   }, [progress]);

   return (
      <Html as="div" fullscreen>
         <div className={s.canvasLoader}>
            <div className={s.loader}></div>
            {/* <p className={s.title}>{progress !== 0 ? `${progress.toFixed(2)}%` : 'Loading...'}</p> */}
         </div>
      </Html>
   );
};

export default CanvasLoader;
