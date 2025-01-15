import React from 'react';
import EmailEditor from './components/email-editor/EmailEditor';
import EmailList from './components/email-list/EmailList';

const Home = () => {
   return (
      <div className="main">
         <EmailEditor />
         <EmailList />
      </div>
   );
};

export default Home;
