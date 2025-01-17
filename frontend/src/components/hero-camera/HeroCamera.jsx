import { Html } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { easing } from 'maath';
import React, { useEffect, useRef, useState } from 'react';
import s from './styles.module.scss';

const HeroCamera = ({ children, isAnimEnded }) => {
   const groupRef = useRef(null);

   useFrame((state, delta) => {
      // easing.damp3(state.camera.position, [0, 0, 20], 0.25, delta);
      if (isAnimEnded) {
         easing.dampE(
            groupRef.current.rotation,
            [state.pointer.y / 6, -state.pointer.x / 8, 0],
            0.4,
            delta,
         );
      }
   });
   return <group ref={groupRef}>{children}</group>;
};

export default HeroCamera;
