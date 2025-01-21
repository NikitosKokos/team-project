import React, { useEffect, useRef, useState } from 'react';
import s from './styles.module.scss';

const ChooseStudent = ({ setIsUserChosen }) => {
   const [students, setStudents] = useState([
      { id: 1, firstName: 'Mykyta', lastName: 'Tsykunov', isSelected: false },
      { id: 2, firstName: 'Cezar', lastName: 'Sîrbu', isSelected: false },
      { id: 3, firstName: 'Danylo', lastName: 'Bordunov', isSelected: false },
      { id: 4, firstName: 'Danylo', lastName: 'Bordunov', isSelected: false },
      { id: 5, firstName: 'Danylo', lastName: 'Bordunov', isSelected: false },
      { id: 6, firstName: 'Danylo', lastName: 'Bordunov', isSelected: false },
   ]);
   const [searchTerm, setSearchTerm] = useState('');
   const [title, setTitle] = useState('Choose a student');
   const [isDropdownOpen, setIsDropdownOpen] = useState(false);
   const [isNewStudentDropdownOpen, setNewStudentIsDropdownOpen] = useState(false);
   const [firstName, setFirstName] = useState('');
   const [lastName, setLastName] = useState('');
   const selectRef = useRef(null); // Reference to the dropdown container

   const filteredStudents = students.filter((student) =>
      `${student.firstName} ${student.lastName}`.toLowerCase().includes(searchTerm.toLowerCase()),
   );

   const handleAddStudent = (e) => {
      e.preventDefault();
      if (firstName.trim() !== '' && lastName.trim() !== '') {
         setStudents([
            ...students,
            { id: Date.now(), firstName: firstName.trim(), lastName: lastName.trim() },
         ]);
         setSearchTerm('');
         setNewStudentIsDropdownOpen(false);
      }
   };

   const handleSelectClick = () => {
      setIsDropdownOpen((prev) => !prev);
      if (!isDropdownOpen) {
         setSearchTerm('');
      }
   };

   const handleClickOutside = (event) => {
      // Check if the click is outside the dropdown and not on a child element
      if (selectRef.current && !selectRef.current.contains(event.target)) {
         setIsDropdownOpen(false);
         setNewStudentIsDropdownOpen(false);
      }
   };

   const handleSelectStudent = ({ firstName, lastName, id }) => {
      setTitle(`${firstName} ${lastName}`);
      const newStudents = students.map((el) => ({
         ...el,
         isSelected: el.id === id,
      }));
      setStudents(newStudents);
      setIsUserChosen(true);
      setIsDropdownOpen(false);
   };

   const handleNewStudent = () => {
      setNewStudentIsDropdownOpen(true);
      setFirstName('');
      setLastName('');
   };

   const handleBack = (e) => {
      e.preventDefault();
      setSearchTerm('');
      setNewStudentIsDropdownOpen(false);
   };

   useEffect(() => {
      document.addEventListener('mousedown', handleClickOutside);
      return () => {
         document.removeEventListener('mousedown', handleClickOutside);
      };
   }, []);

   return (
      <div className={s.chooseStudent}>
         <div className={s.chooseStudent__main}>
            <div className={s.chooseStudent__select} ref={selectRef}>
               <div className={s.chooseStudent__wrapper}>
                  <div className={s.chooseStudent__label}>
                     <div
                        className={`${s.chooseStudent__text} ${isDropdownOpen ? s.active : ''}`}
                        onClick={handleSelectClick}>
                        <span>{title}</span>
                        <svg viewBox="0 0 16 10" fill="none" xmlns="http://www.w3.org/2000/svg">
                           <path
                              d="M7.19313 9.63386C7.63941 10.122 8.36416 10.122 8.81044 9.63386L15.6653 2.13533C16.1116 1.64714 16.1116 0.854325 15.6653 0.366139C15.219 -0.122046 14.4943 -0.122046 14.048 0.366139L8 6.98204L1.95202 0.370045C1.50575 -0.118141 0.780988 -0.118141 0.334709 0.370045C-0.11157 0.858231 -0.11157 1.65104 0.334709 2.13923L7.18956 9.63777L7.19313 9.63386Z"
                              fill="#565356"
                           />
                        </svg>
                     </div>
                     <div
                        className={`${s.chooseStudent__search} ${isDropdownOpen ? s.active : ''}`}>
                        <input
                           type="text"
                           placeholder="Search for a student..."
                           value={searchTerm}
                           onChange={(e) => setSearchTerm(e.target.value)}
                           className={s.chooseStudent__input}
                        />
                     </div>
                  </div>
                  <div
                     className={`${s.chooseStudent__dropdown} ${
                        isDropdownOpen && !isNewStudentDropdownOpen ? s.active : ''
                     }`}>
                     <div className={s.chooseStudent__list}>
                        {filteredStudents.length > 0 ? (
                           filteredStudents.map((student) => (
                              <div
                                 key={student.id}
                                 className={`${s.chooseStudent__item} ${
                                    student.isSelected ? s.active : ''
                                 }`}
                                 onClick={() => handleSelectStudent(student)}>
                                 {student.firstName} {student.lastName}
                              </div>
                           ))
                        ) : (
                           <div className={s.chooseStudent__empty}>No students found</div>
                        )}
                     </div>
                     <button onClick={handleNewStudent} className={s.chooseStudent__add}>
                        <span></span>Add a new student
                     </button>
                  </div>
                  <form
                     className={`${s.chooseStudent__new} ${
                        isNewStudentDropdownOpen ? s.active : ''
                     }`}>
                     <div className={s.chooseStudent__labelNew}>Add a new student</div>
                     <div className={s.chooseStudent__inputs}>
                        <div className={s.chooseStudent__inputNewWrapper}>
                           <input
                              type="text"
                              placeholder="First Name"
                              value={firstName}
                              onChange={(e) => setFirstName(e.target.value)}
                              className={s.chooseStudent__inputNew}
                           />
                        </div>
                        <div className={s.chooseStudent__inputNewWrapper}>
                           <input
                              type="text"
                              placeholder="Second Name"
                              value={lastName}
                              onChange={(e) => setLastName(e.target.value)}
                              className={s.chooseStudent__inputNew}
                           />
                        </div>
                     </div>
                     <div className={s.chooseStudent__buttons}>
                        <button
                           type="submit"
                           onClick={handleAddStudent}
                           className={s.chooseStudent__btn}>
                           Save
                        </button>
                        <button onClick={handleBack} className={s.chooseStudent__btnBack}>
                           Back
                        </button>
                     </div>
                  </form>
               </div>
            </div>
         </div>
      </div>
   );
};

export default ChooseStudent;
