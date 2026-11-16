import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import { PlusCircle, History, PanelLeftClose, PanelLeftOpen } from './icons';

/**
 * Collapsible Sidebar Navigation for Verity.
 * - Collapsed: shows only icons with tooltips.
 * - Expanded: shows black logo, brand name, labels, and tagline.
 * - Monochromatic black & white modern design with smooth CSS transitions.
 */
const Sidebar: React.FC = () => {
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    try {
      return localStorage.getItem('verity_sidebar_collapsed') === 'true';
    } catch {
      return false;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem('verity_sidebar_collapsed', String(collapsed));
    } catch {
      // ignore
    }
  }, [collapsed]);

  const toggle = () => setCollapsed((prev) => !prev);

  return (
    <aside
      style={{
        ...styles.sidebar,
        width: collapsed ? '72px' : '260px',
        minWidth: collapsed ? '72px' : '260px',
      }}
    >
      {/* Brand Header */}
      <div
        style={{
          ...styles.header,
          justifyContent: collapsed ? 'center' : 'space-between',
          padding: collapsed ? '18px 12px' : '18px 16px',
        }}
      >
        <div style={styles.brandRow}>
          <img
            src="/V_Black.png"
            alt="Verity Logo"
            style={styles.logoImg}
          />
          {!collapsed && (
            <div style={styles.brandTextGroup}>
              <span style={styles.brandName}>Verity</span>
              <span style={styles.brandBadge}>EVAL</span>
            </div>
          )}
        </div>

        <button
          onClick={toggle}
          style={styles.toggleBtn}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
        </button>
      </div>

      {/* Navigation */}
      <nav style={styles.nav}>
        <NavLink
          to="/evaluate"
          title={collapsed ? 'New Evaluation' : undefined}
          style={({ isActive }) => ({
            ...styles.navLink,
            ...(collapsed ? styles.navLinkCollapsed : {}),
            ...(isActive ? styles.navLinkActive : styles.navLinkInactive),
          })}
        >
          {({ isActive }) => (
            <>
              <span
                style={{
                  ...styles.iconWrapper,
                  color: isActive ? '#ffffff' : '#09090b',
                }}
              >
                <PlusCircle size={20} strokeWidth={2.2} />
              </span>
              {!collapsed && (
                <span style={styles.navText}>New Evaluation</span>
              )}
            </>
          )}
        </NavLink>

        <NavLink
          to="/history"
          title={collapsed ? 'Evaluation History' : undefined}
          style={({ isActive }) => ({
            ...styles.navLink,
            ...(collapsed ? styles.navLinkCollapsed : {}),
            ...(isActive ? styles.navLinkActive : styles.navLinkInactive),
          })}
        >
          {({ isActive }) => (
            <>
              <span
                style={{
                  ...styles.iconWrapper,
                  color: isActive ? '#ffffff' : '#09090b',
                }}
              >
                <History size={20} strokeWidth={2.2} />
              </span>
              {!collapsed && (
                <span style={styles.navText}>Evaluation History</span>
              )}
            </>
          )}
        </NavLink>
      </nav>

      {/* Footer Tagline */}
      {!collapsed && (
        <div style={styles.footer}>
          <div style={styles.footerDivider} />
          <p style={styles.tagline}>
            Making AI responses more trustworthy, one evaluation at a time.
          </p>
        </div>
      )}
    </aside>
  );
};

const styles: Record<string, React.CSSProperties> = {
  sidebar: {
    height: '100%',
    backgroundColor: '#ffffff',
    borderRight: '1px solid #e4e4e7',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    transition: 'width 0.22s cubic-bezier(0.4, 0, 0.2, 1), min-width 0.22s cubic-bezier(0.4, 0, 0.2, 1)',
    zIndex: 10,
    flexShrink: 0,
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    borderBottom: '1px solid #e4e4e7',
    minHeight: '68px',
  },
  brandRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    overflow: 'hidden',
  },
  logoImg: {
    width: '32px',
    height: '32px',
    objectFit: 'contain',
    flexShrink: 0,
    display: 'block',
  },
  brandTextGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    whiteSpace: 'nowrap',
  },
  brandName: {
    fontSize: '18px',
    fontWeight: '800',
    color: '#09090b',
    letterSpacing: '-0.4px',
  },
  brandBadge: {
    fontSize: '9px',
    fontWeight: '800',
    letterSpacing: '0.8px',
    color: '#09090b',
    backgroundColor: '#f4f4f5',
    border: '1px solid #d4d4d8',
    borderRadius: '4px',
    padding: '1px 5px',
  },
  toggleBtn: {
    background: 'none',
    border: '1px solid #e4e4e7',
    borderRadius: '6px',
    color: '#09090b',
    cursor: 'pointer',
    padding: '6px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'background-color 0.15s, border-color 0.15s',
    flexShrink: 0,
  },
  nav: {
    display: 'flex',
    flexDirection: 'column',
    padding: '14px 10px',
    gap: '6px',
    flex: 1,
  },
  navLink: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '10px 14px',
    borderRadius: '8px',
    fontSize: '15px',
    fontWeight: '600',
    textDecoration: 'none',
    transition: 'all 0.15s ease',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
  },
  navLinkCollapsed: {
    justifyContent: 'center',
    padding: '12px 0',
  },
  navLinkActive: {
    backgroundColor: '#09090b',
    color: '#ffffff',
  },
  navLinkInactive: {
    backgroundColor: 'transparent',
    color: '#52525b',
  },
  iconWrapper: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  navText: {
    fontSize: '15px',
    letterSpacing: '-0.2px',
  },
  footer: {
    padding: '16px',
  },
  footerDivider: {
    height: '1px',
    backgroundColor: '#e4e4e7',
    marginBottom: '12px',
  },
  tagline: {
    fontSize: '12px',
    color: '#71717a',
    lineHeight: '1.5',
    fontWeight: '500',
  },
};

export default Sidebar;
