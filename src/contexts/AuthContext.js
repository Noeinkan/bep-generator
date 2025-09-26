import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const storedUser = localStorage.getItem('currentUser');
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
    setLoading(false);
  }, []);

  const register = async (userData) => {
    try {
      const { email, password, name } = userData;

      const existingUsers = JSON.parse(localStorage.getItem('users') || '[]');

      if (existingUsers.find(u => u.email === email)) {
        throw new Error('User with this email already exists');
      }

      const newUser = {
        id: Date.now().toString(),
        email,
        name,
        password,
        createdAt: new Date().toISOString(),
        projects: []
      };

      const updatedUsers = [...existingUsers, newUser];
      localStorage.setItem('users', JSON.stringify(updatedUsers));

      const userForStorage = { ...newUser };
      delete userForStorage.password;

      setUser(userForStorage);
      localStorage.setItem('currentUser', JSON.stringify(userForStorage));

      return { success: true, user: userForStorage };
    } catch (error) {
      return { success: false, error: error.message };
    }
  };

  const login = async (credentials) => {
    try {
      const { email, password } = credentials;
      const existingUsers = JSON.parse(localStorage.getItem('users') || '[]');

      const user = existingUsers.find(u => u.email === email);

      if (!user || user.password !== password) {
        throw new Error('Invalid email or password');
      }

      const userForStorage = { ...user };
      delete userForStorage.password;

      setUser(userForStorage);
      localStorage.setItem('currentUser', JSON.stringify(userForStorage));

      return { success: true, user: userForStorage };
    } catch (error) {
      return { success: false, error: error.message };
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('currentUser');
  };

  const updateUserProjects = (projects) => {
    if (!user) return;

    const updatedUser = { ...user, projects };
    setUser(updatedUser);
    localStorage.setItem('currentUser', JSON.stringify(updatedUser));

    const existingUsers = JSON.parse(localStorage.getItem('users') || '[]');
    const updatedUsers = existingUsers.map(u =>
      u.id === user.id ? { ...u, projects } : u
    );
    localStorage.setItem('users', JSON.stringify(updatedUsers));
  };

  const value = {
    user,
    loading,
    register,
    login,
    logout,
    updateUserProjects,
    isAuthenticated: !!user
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};