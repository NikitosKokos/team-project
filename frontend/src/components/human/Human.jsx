import React, { useEffect, useRef } from 'react';
import { useGLTF } from '@react-three/drei';
import { MeshStandardMaterial } from 'three';

const Human = (props) => {
   const humanRef = useRef(null);
   const { nodes, materials } = useGLTF('/models/wireframe_man.glb');

   return (
      <group {...props} dispose={null} ref={humanRef}>
         <group scale={0.01}>
            <group
               position={[0, -10000, 0]}
               rotation={[-Math.PI / 2, 0, Math.PI / 2]}
               scale={10000}>
               <primitive object={nodes._rootJoint} />
               <skinnedMesh
                  geometry={nodes.Object_7.geometry}
                  // material={materials['Material.001']}
                  skeleton={nodes.Object_7.skeleton}
                  material={new MeshStandardMaterial({ color: '#2C919D' })} // Use custom material
               />
            </group>
         </group>
      </group>
   );
};

useGLTF.preload('/models/wireframe_man.glb');

export default Human;
