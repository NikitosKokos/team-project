import React, { useState } from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router';
import Login from '../login/Login.jsx';
import Overview from '../overview/Overview.jsx';
import MainPage from '../main-page/MainPage.jsx';

const App = () => {
   const [isLoggedIn, setIsLoggedIn] = useState(true);
   return (
      <BrowserRouter>
         <Routes>
            <Route path="/*" element={isLoggedIn ? <MainPage /> : <Navigate to="/login" />} />
            <Route path="/login" element={<Login />} />
            <Route path="/overview" element={<Overview />} />
         </Routes>
      </BrowserRouter>
   );
};

export default App;
