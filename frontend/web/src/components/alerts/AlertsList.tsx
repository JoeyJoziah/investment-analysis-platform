import React, { memo } from 'react';
import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Chip,
  Tooltip,
  Button,
  Menu,
  MenuItem,
} from '@mui/material';
import {
  Add,
  Delete,
  Edit,
  MoreVert,
  NotificationsActive,
  NotificationsOff,
  TrendingUp,
  TrendingDown,
  VolumeUp,
  Newspaper,
  PieChart,
} from '@mui/icons-material';
import type { AlertType, AlertCondition } from './AlertForm';

// --- Types ---

export type AlertStatus = 'active' | 'triggered' | 'expired' | 'paused';

export interface AlertItem {
  id: string;
  ticker: string;
  type: AlertType;
  condition: AlertCondition;
  value: number;
  status: AlertStatus;
  notifyInApp: boolean;
  notifyEmail: boolean;
  createdAt: string;
  triggeredAt: string | null;
  expiresAt: string | null;
  message: string;
}

// --- Constants ---

const TYPE_LABELS: Record<AlertType, string> = {
  price: 'Price',
  volume: 'Volume',
  percent_change: '% Change',
  news: 'News',
  portfolio_drift: 'Portfolio Drift',
};

const TYPE_ICONS: Record<AlertType, React.ReactElement> = {
  price: <TrendingUp fontSize="small" />,
  volume: <VolumeUp fontSize="small" />,
  percent_change: <TrendingDown fontSize="small" />,
  news: <Newspaper fontSize="small" />,
  portfolio_drift: <PieChart fontSize="small" />,
};

const STATUS_COLORS: Record<AlertStatus, 'success' | 'warning' | 'error' | 'default'> = {
  active: 'success',
  triggered: 'warning',
  expired: 'error',
  paused: 'default',
};

// --- Helpers ---

export function formatDate(iso: string | null): string {
  if (!iso) return '-';
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function formatAlertValue(type: AlertType, value: number): string {
  if (type === 'price') {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  }
  if (type === 'percent_change' || type === 'portfolio_drift') {
    return `${value >= 0 ? '+' : ''}${value}%`;
  }
  if (type === 'volume') {
    if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(0)}K`;
    return value.toFixed(0);
  }
  return String(value);
}

export function isExpiringSoon(expiresAt: string | null): boolean {
  if (!expiresAt) return false;
  const diff = new Date(expiresAt).getTime() - Date.now();
  return diff > 0 && diff < 7 * 24 * 60 * 60 * 1000;
}

export function buildMessage(
  type: AlertType,
  ticker: string,
  condition: AlertCondition,
  value: number
): string {
  if (type === 'news') return `${ticker} breaking news alert`;
  if (type === 'portfolio_drift') {
    return `Portfolio drift exceeds ${formatAlertValue(type, value)}`;
  }
  return `${ticker} ${TYPE_LABELS[type].toLowerCase()} ${condition} ${formatAlertValue(type, value)}`;
}

// --- Small presentation helpers ---

const NotifChips: React.FC<{ inApp: boolean; email: boolean }> = ({ inApp, email }) => (
  <Box sx={{ display: 'flex', justifyContent: 'center', gap: 0.5 }}>
    {inApp && <Chip label="App" size="small" variant="outlined" />}
    {email && <Chip label="Email" size="small" variant="outlined" />}
  </Box>
);

const TypeCell: React.FC<{ type: AlertType }> = ({ type }) => (
  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
    {TYPE_ICONS[type]}
    <Typography variant="body2">{TYPE_LABELS[type]}</Typography>
  </Box>
);

// --- Props ---

interface AlertsListProps {
  alerts: AlertItem[];
  tabValue: number;
  searchQuery: string;
  filterType: AlertType | 'all';
  anchorEl: HTMLElement | null;
  menuId: string | null;
  onToggle: (id: string) => void;
  onDelete: (id: string) => void;
  onEdit: (alert: AlertItem) => void;
  onMenuOpen: (e: React.MouseEvent<HTMLElement>, id: string) => void;
  onMenuClose: () => void;
  onCreateOpen: () => void;
}

/**
 * AlertsList - Renders the active-alerts or history table depending on the
 * currently selected tab. Includes per-row actions (toggle, context menu)
 * and empty-state messaging.
 */
const AlertsList: React.FC<AlertsListProps> = ({
  alerts,
  tabValue,
  searchQuery,
  filterType,
  anchorEl,
  menuId,
  onToggle,
  onDelete,
  onEdit,
  onMenuOpen,
  onMenuClose,
  onCreateOpen,
}) => {
  const menuAlert = alerts.find((a) => a.id === menuId);

  // Active alerts tab
  if (tabValue === 0) {
    return (
      <>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Ticker</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Condition</TableCell>
                <TableCell align="right">Value</TableCell>
                <TableCell align="center">Status</TableCell>
                <TableCell align="center">Notifications</TableCell>
                <TableCell>Created</TableCell>
                <TableCell>Expires</TableCell>
                <TableCell align="center">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {alerts.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={9} align="center" sx={{ py: 6 }}>
                    <Typography color="text.secondary">
                      {searchQuery || filterType !== 'all'
                        ? 'No alerts match your filters'
                        : 'No active alerts. Create one to get started.'}
                    </Typography>
                    {!searchQuery && filterType === 'all' && (
                      <Button
                        variant="contained"
                        sx={{ mt: 2 }}
                        startIcon={<Add />}
                        onClick={onCreateOpen}
                      >
                        Create Your First Alert
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              ) : (
                alerts.map((a) => (
                  <TableRow key={a.id} hover>
                    <TableCell>
                      <Typography fontWeight="bold">{a.ticker}</Typography>
                    </TableCell>
                    <TableCell>
                      <TypeCell type={a.type} />
                    </TableCell>
                    <TableCell>
                      <Typography
                        variant="body2"
                        sx={{ textTransform: 'capitalize' }}
                      >
                        {a.type === 'news' ? '-' : a.condition}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Typography variant="body2">
                        {a.type === 'news'
                          ? '-'
                          : formatAlertValue(a.type, a.value)}
                      </Typography>
                    </TableCell>
                    <TableCell align="center">
                      <Chip
                        label={
                          a.status.charAt(0).toUpperCase() + a.status.slice(1)
                        }
                        color={STATUS_COLORS[a.status]}
                        size="small"
                      />
                    </TableCell>
                    <TableCell align="center">
                      <NotifChips
                        inApp={a.notifyInApp}
                        email={a.notifyEmail}
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {formatDate(a.createdAt)}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography
                        variant="body2"
                        color={
                          isExpiringSoon(a.expiresAt)
                            ? 'warning.main'
                            : 'text.secondary'
                        }
                      >
                        {a.expiresAt ? formatDate(a.expiresAt) : 'Never'}
                      </Typography>
                    </TableCell>
                    <TableCell align="center">
                      <Box
                        sx={{ display: 'flex', justifyContent: 'center' }}
                      >
                        <Tooltip
                          title={
                            a.status === 'active'
                              ? 'Pause alert'
                              : a.status === 'paused'
                                ? 'Resume alert'
                                : 'Cannot toggle'
                          }
                        >
                          <span>
                            <IconButton
                              size="small"
                              color={
                                a.status === 'active' ? 'primary' : 'default'
                              }
                              onClick={() => onToggle(a.id)}
                              disabled={
                                a.status === 'expired' ||
                                a.status === 'triggered'
                              }
                            >
                              {a.status === 'active' ? (
                                <NotificationsActive fontSize="small" />
                              ) : (
                                <NotificationsOff fontSize="small" />
                              )}
                            </IconButton>
                          </span>
                        </Tooltip>
                        <IconButton
                          size="small"
                          onClick={(e) => onMenuOpen(e, a.id)}
                        >
                          <MoreVert fontSize="small" />
                        </IconButton>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>

        {/* Context Menu */}
        <Menu
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={onMenuClose}
        >
          <MenuItem
            onClick={() => {
              if (menuAlert) onEdit(menuAlert);
            }}
          >
            <Edit sx={{ mr: 1 }} fontSize="small" /> Edit
          </MenuItem>
          <MenuItem
            onClick={() => {
              if (menuId) {
                onToggle(menuId);
                onMenuClose();
              }
            }}
          >
            <NotificationsOff sx={{ mr: 1 }} fontSize="small" />
            {menuAlert?.status === 'active' ? 'Pause' : 'Resume'}
          </MenuItem>
          <MenuItem
            onClick={() => {
              if (menuId) {
                onDelete(menuId);
                onMenuClose();
              }
            }}
          >
            <Delete sx={{ mr: 1 }} fontSize="small" color="error" /> Delete
          </MenuItem>
        </Menu>
      </>
    );
  }

  // Alert history tab
  return (
    <TableContainer>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Ticker</TableCell>
            <TableCell>Type</TableCell>
            <TableCell>Alert</TableCell>
            <TableCell>Triggered At</TableCell>
            <TableCell align="center">Notifications Sent</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {alerts.length === 0 ? (
            <TableRow>
              <TableCell colSpan={5} align="center" sx={{ py: 6 }}>
                <Typography color="text.secondary">
                  No triggered alerts to display
                </Typography>
              </TableCell>
            </TableRow>
          ) : (
            alerts.map((a) => (
              <TableRow key={a.id} hover>
                <TableCell>
                  <Typography fontWeight="bold">{a.ticker}</Typography>
                </TableCell>
                <TableCell>
                  <TypeCell type={a.type} />
                </TableCell>
                <TableCell>
                  <Typography variant="body2">{a.message}</Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="body2">
                    {formatDate(a.triggeredAt)}
                  </Typography>
                </TableCell>
                <TableCell align="center">
                  <NotifChips inApp={a.notifyInApp} email={a.notifyEmail} />
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default memo(AlertsList);
