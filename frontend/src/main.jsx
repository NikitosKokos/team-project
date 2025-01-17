import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './assets/style/index.scss';
import App from './pages/home/App.jsx';

createRoot(document.getElementById('root')).render(
   <StrictMode>
      <App />
   </StrictMode>,
);
