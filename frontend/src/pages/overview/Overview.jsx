import { Canvas, useFrame } from '@react-three/fiber';
import React, { Suspense, useEffect, useRef, useState } from 'react';
import s from './styles.module.scss';
import { Html, OrbitControls, PerspectiveCamera, Sparkles } from '@react-three/drei';
import Human from '../../components/human/Human';
import CanvasLoader from '../../components/canvas-loader/CanvasLoader';
import { Leva, useControls } from 'leva';
import { useMediaQuery } from 'react-responsive';
import gsap from 'gsap';
import HeroCamera from '../../components/hero-camera/HeroCamera';
import LoginForm from '../../components/login-form/LoginForm';

const Overview = () => {
   const isSmall = useMediaQuery({ maxWidth: 440 });
   const isMobile = useMediaQuery({ maxWidth: 768 });
   const isTablet = useMediaQuery({ maxWidth: 768, maxWidth: 1024 });

   const animDuration = 4;
   let timeout;

   const cameraRef = useRef(null);
   const [isCameraReady, setIsCameraReady] = useState(false);
   const [isAnimEnded, setIsAnimEnded] = useState(false);

   useEffect(() => {
      if (isCameraReady && cameraRef.current) {
         gsap.to(cameraRef.current.position, {
            x: 0,
            y: 0,
            z: 30,
            duration: animDuration,
            ease: 'power1.inOut',
         });
         // Animate rotation
         gsap.to(cameraRef.current.rotation, {
            x: 0,
            y: 0,
            z: 0,
            duration: animDuration,
            ease: 'power1.inOut',
         });

         timeout = setTimeout(() => {
            setIsAnimEnded(true);
         }, animDuration * 1000);
      }
   }, [isCameraReady]);

   useEffect(() => {
      return () => {
         clearTimeout(timeout);
      };
   }, []);

   const handleCameraReady = (camera) => {
      cameraRef.current = camera; // Assign the camera instance
      setIsCameraReady(true); // Mark the camera as ready
   };
   // const x = useControls('Human', {
   //    positionX: {
   //       value: 2.5,
   //       min: -10,
   //       max: 10,
   //    },
   //    positionY: {
   //       value: 2.5,
   //       min: -100,
   //       max: 100,
   //    },
   //    positionZ: {
   //       value: 2.5,
   //       min: -10,
   //       max: 10,
   //    },
   //    rotationX: {
   //       value: 2.5,
   //       min: -10,
   //       max: 10,
   //    },
   //    rotationY: {
   //       value: 2.5,
   //       min: -10,
   //       max: 10,
   //    },
   //    rotationZ: {
   //       value: 2.5,
   //       min: -10,
   //       max: 10,
   //    },
   //    scale: {
   //       value: 2.5,
   //       min: 0.1,
   //       max: 10,
   //    },
   // });
   return (
      <>
         {/* <Leva /> */}
         <div className={s.hero}>
            {/* <LoginForm isAnimEnded={isAnimEnded} /> */}
            <Canvas
               className={s.canvas}
               style={{ height: '100%', width: '100%', overflow: 'hidden' }}>
               <Suspense fallback={<CanvasLoader />}>
                  {/* <CanvasLoader /> */}
                  <PerspectiveCamera
                     makeDefault
                     position={[20, -50, 0]}
                     rotation={[-Math.PI / 2, 0, Math.PI / 4]}
                     onUpdate={handleCameraReady}
                  />
                  <HeroCamera isAnimEnded={isAnimEnded}>
                     <Human
                        // scale={0.05}
                        // scale={[x.scale, x.scale, x.scale]}
                        // position={[x.positionX, x.positionY, x.positionZ]}
                        // rotation={[x.rotationX, x.rotationY, x.rotationZ]}
                        // position={[0, 0, 0]}
                        // rotation={[0, -Math.PI / 2.0, 0]}
                        scale={0.5}
                        position={[0, -40, -10]}
                        rotation={[-3.3, -1.9, -3.3]}
                     />
                  </HeroCamera>
                  <ambientLight intensity={1} />
                  <directionalLight position={[10, 10, 10]} intensity={0.7} />
               </Suspense>
            </Canvas>
         </div>
      </>
   );
};

export default Overview;
