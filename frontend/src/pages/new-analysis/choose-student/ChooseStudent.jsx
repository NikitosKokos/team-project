import React, { useState } from 'react';
import s from './styles.module.scss';

const ChooseStudent = () => {
   const [students, setStudents] = useState([
      // Example initial data, replace with your data
      { id: 1, name: 'John Doe' },
      { id: 2, name: 'Jane Smith' },
   ]);
   const [searchTerm, setSearchTerm] = useState('');
   const [isDropdownOpen, setIsDropdownOpen] = useState(false);
   const [newStudent, setNewStudent] = useState('');

   const filteredStudents = students.filter((student) =>
      student.name.toLowerCase().includes(searchTerm.toLowerCase()),
   );

   const handleAddStudent = () => {
      if (newStudent.trim()) {
         setStudents([...students, { id: Date.now(), name: newStudent.trim() }]);
         setNewStudent('');
         setSearchTerm('');
      }
   };

   return (
      <div className={s.chooseStudent}>
         <div className={s.chooseStudent__title}>Choose a student</div>
         <div className={s.chooseStudent__main}>
            <div
               className={s.chooseStudent__select}
               onClick={() => setIsDropdownOpen(!isDropdownOpen)}>
               <div className={s.chooseStudent__wrapper}>
                  <div className={s.chooseStudent__label}>
                     {isDropdownOpen ? (
                        <>
                        <span>Choose a student</span>
                        <svg viewBox="0 0 16 10" fill="none" xmlns="http://www.w3.org/2000/svg">
                           <path
                              d="M7.19313 9.63386C7.63941 10.122 8.36416 10.122 8.81044 9.63386L15.6653 2.13533C16.1116 1.64714 16.1116 0.854325 15.6653 0.366139C15.219 -0.122046 14.4943 -0.122046 14.048 0.366139L8 6.98204L1.95202 0.370045C1.50575 -0.118141 0.780988 -0.118141 0.334709 0.370045C-0.11157 0.858231 -0.11157 1.65104 0.334709 2.13923L7.18956 9.63777L7.19313 9.63386Z"
                              fill="#565356"
                           />
                        </svg>                        
                        </>
                     )
                  : (
                     
                  )}

                  </div>
                  <div className={`${s.chooseStudent__dropdown} ${isDropdownOpen ? s.active : ''}`}>
                     <div className={s.chooseStudent__search}>
                        <input
                           type="text"
                           placeholder="Search for a student..."
                           value={searchTerm}
                           onChange={(e) => setSearchTerm(e.target.value)}
                           className={s.chooseStudent__input}
                        />
                     </div>
                     <div className={s.chooseStudent__list}>
                        {filteredStudents.length > 0 ? (
                           filteredStudents.map((student) => (
                              <div key={student.id} className={s.chooseStudent__item}>
                                 {student.name}
                              </div>
                           ))
                        ) : (
                           <div className={s.chooseStudent__empty}>No students found</div>
                        )}
                     </div>
                     <div className={s.chooseStudent__add}>
                        <input
                           type="text"
                           placeholder="Add a new student..."
                           value={newStudent}
                           onChange={(e) => setNewStudent(e.target.value)}
                           className={s.chooseStudent__input}
                        />
                        <button className={s.chooseStudent__addButton} onClick={handleAddStudent}>
                           Add
                        </button>
                     </div>
                  </div>
               </div>
            </div>
         </div>
      </div>
   );
};

export default ChooseStudent;
