import React from 'react';
import Sidebar from './Sidebar';

interface LayoutProps {
  children: React.ReactNode;
}

/**
 * Main layout shell: fixed sidebar on the left, scrollable content area on the right.
 */
const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div style={styles.container}>
      <Sidebar />
      <main style={styles.main}>
        {children}
      </main>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    height: '100%',
    width: '100%',
    overflow: 'hidden',
  },
  main: {
    flex: 1,
    overflowY: 'auto',
    backgroundColor: 'var(--bg-primary)',
    minWidth: 0, // prevent flex overflow
  },
};

export default Layout;
