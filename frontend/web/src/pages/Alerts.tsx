import React, { useState, useMemo, useCallback } from 'react';
import {
  Container, Grid, Paper, Typography, Box, Card, CardContent, Table, TableBody,
  TableCell, TableContainer, TableHead, TableRow, Button, IconButton, Chip, Tabs,
  Tab, Tooltip, Dialog, DialogTitle, DialogContent, DialogActions, TextField,
  MenuItem, Switch, FormControlLabel, FormControl, InputLabel, Select,
  InputAdornment, Menu,
} from '@mui/material';
import {
  Add, Delete, Edit, MoreVert, NotificationsActive, NotificationsOff,
  TrendingUp, TrendingDown, VolumeUp, Newspaper, PieChart, FilterList,
  Search, AccessTime,
} from '@mui/icons-material';
import { useAppDispatch } from '../hooks/redux';
import { addNotification } from '../store/slices/appSlice';

// --- Types ---

type AlertType = 'price' | 'volume' | 'percent_change' | 'news' | 'portfolio_drift';
type AlertCondition = 'above' | 'below' | 'equals';
type AlertStatus = 'active' | 'triggered' | 'expired' | 'paused';

interface AlertItem {
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

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

// --- Constants ---

const TYPE_LABELS: Record<AlertType, string> = {
  price: 'Price', volume: 'Volume', percent_change: '% Change',
  news: 'News', portfolio_drift: 'Portfolio Drift',
};

const TYPE_ICONS: Record<AlertType, React.ReactElement> = {
  price: <TrendingUp fontSize="small" />, volume: <VolumeUp fontSize="small" />,
  percent_change: <TrendingDown fontSize="small" />, news: <Newspaper fontSize="small" />,
  portfolio_drift: <PieChart fontSize="small" />,
};

const STATUS_COLORS: Record<AlertStatus, 'success' | 'warning' | 'error' | 'default'> = {
  active: 'success', triggered: 'warning', expired: 'error', paused: 'default',
};

const mkAlert = (
  id: string, ticker: string, type: AlertType, condition: AlertCondition,
  value: number, status: AlertStatus, notifyEmail: boolean, createdAt: string,
  triggeredAt: string | null, expiresAt: string | null, message: string,
): AlertItem => ({
  id, ticker, type, condition, value, status, notifyInApp: true, notifyEmail,
  createdAt, triggeredAt, expiresAt, message,
});

const INITIAL_ALERTS: AlertItem[] = [
  mkAlert('a1','AAPL','price','above',200,'active',false,'2026-02-25T10:30:00Z',null,'2026-04-01T00:00:00Z','AAPL price above $200'),
  mkAlert('a2','TSLA','percent_change','below',-5,'triggered',true,'2026-02-20T08:00:00Z','2026-02-28T14:22:00Z',null,'TSLA dropped more than 5%'),
  mkAlert('a3','GOOGL','volume','above',50000000,'active',false,'2026-02-22T09:15:00Z',null,null,'GOOGL volume above 50M'),
  mkAlert('a4','MSFT','price','below',380,'expired',true,'2026-01-15T12:00:00Z',null,'2026-02-15T00:00:00Z','MSFT price below $380'),
  mkAlert('a5','NVDA','price','above',950,'triggered',true,'2026-02-10T11:00:00Z','2026-03-01T09:31:00Z',null,'NVDA price above $950'),
  mkAlert('a6','AMZN','news','equals',0,'active',false,'2026-02-27T16:00:00Z',null,null,'AMZN breaking news alert'),
  mkAlert('a7','Portfolio','portfolio_drift','above',10,'active',true,'2026-02-18T07:45:00Z',null,null,'Portfolio drift exceeds 10%'),
  mkAlert('a8','META','percent_change','above',3,'paused',false,'2026-02-23T13:30:00Z',null,'2026-03-15T00:00:00Z','META gains more than 3%'),
];

const EMPTY_FORM = {
  ticker: '', type: 'price' as AlertType, condition: 'above' as AlertCondition,
  value: '', notifyInApp: true, notifyEmail: false,
};

// --- Helpers ---

function TabPanel({ children, value, index, ...other }: TabPanelProps) {
  return (
    <div hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

function formatDate(iso: string | null): string {
  if (!iso) return '-';
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit',
  });
}

function formatAlertValue(type: AlertType, value: number): string {
  if (type === 'price') return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);
  if (type === 'percent_change' || type === 'portfolio_drift') return `${value >= 0 ? '+' : ''}${value}%`;
  if (type === 'volume') {
    if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(0)}K`;
    return value.toFixed(0);
  }
  return String(value);
}

function buildMessage(type: AlertType, ticker: string, condition: AlertCondition, value: number): string {
  if (type === 'news') return `${ticker} breaking news alert`;
  if (type === 'portfolio_drift') return `Portfolio drift exceeds ${formatAlertValue(type, value)}`;
  return `${ticker} ${TYPE_LABELS[type].toLowerCase()} ${condition} ${formatAlertValue(type, value)}`;
}

function isExpiringSoon(expiresAt: string | null): boolean {
  if (!expiresAt) return false;
  const diff = new Date(expiresAt).getTime() - Date.now();
  return diff > 0 && diff < 7 * 24 * 60 * 60 * 1000;
}

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

// --- Component ---

const Alerts: React.FC = () => {
  const dispatch = useAppDispatch();
  const [alerts, setAlerts] = useState<AlertItem[]>(INITIAL_ALERTS);
  const [tabValue, setTabValue] = useState(0);
  const [createOpen, setCreateOpen] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [form, setForm] = useState(EMPTY_FORM);
  const [editingAlert, setEditingAlert] = useState<AlertItem | null>(null);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [menuId, setMenuId] = useState<string | null>(null);
  const [filterType, setFilterType] = useState<AlertType | 'all'>('all');
  const [searchQuery, setSearchQuery] = useState('');

  // Derived data
  const activeAlerts = useMemo(() => alerts.filter((a) => a.status === 'active'), [alerts]);
  const triggeredToday = useMemo(() => {
    const start = new Date(); start.setHours(0, 0, 0, 0);
    return alerts.filter((a) => a.status === 'triggered' && a.triggeredAt && new Date(a.triggeredAt) >= start);
  }, [alerts]);
  const expiringSoon = useMemo(() => alerts.filter((a) => a.status === 'active' && isExpiringSoon(a.expiresAt)), [alerts]);

  const filteredAlerts = useMemo(() => {
    const base = tabValue === 0
      ? alerts.filter((a) => a.status !== 'triggered')
      : alerts.filter((a) => a.status === 'triggered');
    return base
      .filter((a) => filterType === 'all' || a.type === filterType)
      .filter((a) => a.ticker.toLowerCase().includes(searchQuery.toLowerCase()) || a.message.toLowerCase().includes(searchQuery.toLowerCase()));
  }, [alerts, tabValue, filterType, searchQuery]);

  // Handlers
  const notify = useCallback((type: 'success' | 'info' | 'error', message: string) => {
    dispatch(addNotification({ type, message }));
  }, [dispatch]);

  const handleCreate = useCallback(() => {
    if (!form.ticker.trim() || (form.type !== 'news' && !form.value)) return;
    const val = form.type === 'news' ? 0 : Number(form.value);
    const ticker = form.ticker.toUpperCase();
    const newAlert: AlertItem = {
      id: `alert-${Date.now()}`, ticker, type: form.type, condition: form.condition,
      value: val, status: 'active', notifyInApp: form.notifyInApp, notifyEmail: form.notifyEmail,
      createdAt: new Date().toISOString(), triggeredAt: null, expiresAt: null,
      message: buildMessage(form.type, ticker, form.condition, val),
    };
    setAlerts((prev) => [...prev, newAlert]);
    setForm(EMPTY_FORM);
    setCreateOpen(false);
    notify('success', `Alert created for ${ticker}`);
  }, [form, notify]);

  const handleEdit = useCallback(() => {
    if (!editingAlert) return;
    const val = form.type === 'news' ? 0 : Number(form.value);
    const ticker = form.ticker.toUpperCase();
    setAlerts((prev) => prev.map((a) => a.id === editingAlert.id ? {
      ...a, ticker, type: form.type, condition: form.condition, value: val,
      notifyInApp: form.notifyInApp, notifyEmail: form.notifyEmail,
      message: buildMessage(form.type, ticker, form.condition, val),
    } : a));
    setEditOpen(false);
    setEditingAlert(null);
    setForm(EMPTY_FORM);
    notify('success', `Alert updated for ${ticker}`);
  }, [editingAlert, form, notify]);

  const handleDelete = useCallback((id: string) => {
    const target = alerts.find((a) => a.id === id);
    setAlerts((prev) => prev.filter((a) => a.id !== id));
    notify('info', `Alert removed${target ? ` for ${target.ticker}` : ''}`);
  }, [alerts, notify]);

  const handleToggle = useCallback((id: string) => {
    const target = alerts.find((a) => a.id === id);
    setAlerts((prev) => prev.map((a) => {
      if (a.id !== id) return a;
      return { ...a, status: (a.status === 'active' ? 'paused' : 'active') as AlertStatus };
    }));
    if (target) notify('info', `Alert ${target.status === 'active' ? 'paused' : 'resumed'} for ${target.ticker}`);
  }, [alerts, notify]);

  const openEdit = useCallback((alert: AlertItem) => {
    setEditingAlert(alert);
    setForm({ ticker: alert.ticker, type: alert.type, condition: alert.condition,
      value: alert.type === 'news' ? '' : String(alert.value),
      notifyInApp: alert.notifyInApp, notifyEmail: alert.notifyEmail });
    setEditOpen(true);
    setAnchorEl(null);
    setMenuId(null);
  }, []);

  const handleMenuOpen = useCallback((e: React.MouseEvent<HTMLElement>, id: string) => {
    setAnchorEl(e.currentTarget); setMenuId(id);
  }, []);
  const handleMenuClose = useCallback(() => { setAnchorEl(null); setMenuId(null); }, []);

  const adornment = useMemo(() => {
    if (form.type === 'price') return { start: '$' };
    if (form.type === 'percent_change' || form.type === 'portfolio_drift') return { end: '%' };
    return {};
  }, [form.type]);

  const formDisabled = !form.ticker.trim() || (form.type !== 'news' && !form.value);
  const menuAlert = alerts.find((a) => a.id === menuId);

  // Shared form renderer
  const renderForm = () => (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: 1 }}>
      <TextField autoFocus label="Ticker Symbol" fullWidth value={form.ticker}
        onChange={(e) => setForm({ ...form, ticker: e.target.value.toUpperCase() })}
        disabled={form.type === 'portfolio_drift'}
        helperText={form.type === 'portfolio_drift' ? 'Portfolio-level alert' : undefined} />
      <FormControl fullWidth>
        <InputLabel>Alert Type</InputLabel>
        <Select value={form.type} label="Alert Type" onChange={(e) => {
          const t = e.target.value as AlertType;
          setForm({ ...form, type: t, ticker: t === 'portfolio_drift' ? 'Portfolio' : form.ticker,
            condition: t === 'news' ? 'equals' : form.condition });
        }}>
          <MenuItem value="price">Price</MenuItem>
          <MenuItem value="volume">Volume</MenuItem>
          <MenuItem value="percent_change">Percentage Change</MenuItem>
          <MenuItem value="news">News</MenuItem>
          <MenuItem value="portfolio_drift">Portfolio Drift</MenuItem>
        </Select>
      </FormControl>
      {form.type !== 'news' && (
        <>
          <FormControl fullWidth>
            <InputLabel>Condition</InputLabel>
            <Select value={form.condition} label="Condition"
              onChange={(e) => setForm({ ...form, condition: e.target.value as AlertCondition })}>
              <MenuItem value="above">Above</MenuItem>
              <MenuItem value="below">Below</MenuItem>
              <MenuItem value="equals">Equals</MenuItem>
            </Select>
          </FormControl>
          <TextField label="Value" type="number" fullWidth value={form.value}
            onChange={(e) => setForm({ ...form, value: e.target.value })}
            InputProps={{
              startAdornment: adornment.start ? <InputAdornment position="start">{adornment.start}</InputAdornment> : undefined,
              endAdornment: adornment.end ? <InputAdornment position="end">{adornment.end}</InputAdornment> : undefined,
            }} />
        </>
      )}
      <Typography variant="subtitle2" sx={{ mt: 1 }}>Notification Method</Typography>
      <Box sx={{ display: 'flex', gap: 3 }}>
        <FormControlLabel control={<Switch checked={form.notifyInApp} onChange={(e) => setForm({ ...form, notifyInApp: e.target.checked })} />} label="In-App" />
        <FormControlLabel control={<Switch checked={form.notifyEmail} onChange={(e) => setForm({ ...form, notifyEmail: e.target.checked })} />} label="Email" />
      </Box>
    </Box>
  );

  // Summary card helper
  const SummaryCard: React.FC<{ label: string; count: number; color: string; icon: React.ReactNode }> = ({ label, count, color, icon }) => (
    <Grid item xs={12} sm={4}>
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Box>
              <Typography color="text.secondary" gutterBottom variant="caption">{label}</Typography>
              <Typography variant="h4" fontWeight="bold" color={`${color}.main`}>{count}</Typography>
            </Box>
            {icon}
          </Box>
        </CardContent>
      </Card>
    </Grid>
  );

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h4" fontWeight="bold">Alerts</Typography>
            <Typography variant="body2" color="text.secondary">Monitor price movements, volume spikes, and portfolio drift</Typography>
          </Box>
          <Button variant="contained" startIcon={<Add />} onClick={() => { setForm(EMPTY_FORM); setCreateOpen(true); }}>
            Create Alert
          </Button>
        </Box>

        {/* Summary Cards */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <SummaryCard label="Total Active" count={activeAlerts.length} color="success" icon={<NotificationsActive sx={{ fontSize: 40, color: 'success.main', opacity: 0.3 }} />} />
          <SummaryCard label="Triggered Today" count={triggeredToday.length} color="warning" icon={<TrendingUp sx={{ fontSize: 40, color: 'warning.main', opacity: 0.3 }} />} />
          <SummaryCard label="Expiring Soon" count={expiringSoon.length} color="error" icon={<AccessTime sx={{ fontSize: 40, color: 'error.main', opacity: 0.3 }} />} />
        </Grid>

        {/* Search and Filter */}
        <Paper sx={{ p: 2, mb: 3 }}>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
            <TextField placeholder="Search alerts..." value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)} sx={{ minWidth: 250, flexGrow: 1 }}
              InputProps={{ startAdornment: <InputAdornment position="start"><Search /></InputAdornment> }} />
            <FormControl sx={{ minWidth: 160 }}>
              <InputLabel>Alert Type</InputLabel>
              <Select value={filterType} label="Alert Type" onChange={(e) => setFilterType(e.target.value as AlertType | 'all')}
                startAdornment={<InputAdornment position="start"><FilterList /></InputAdornment>}>
                <MenuItem value="all">All Types</MenuItem>
                <MenuItem value="price">Price</MenuItem>
                <MenuItem value="volume">Volume</MenuItem>
                <MenuItem value="percent_change">% Change</MenuItem>
                <MenuItem value="news">News</MenuItem>
                <MenuItem value="portfolio_drift">Portfolio Drift</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </Paper>

        {/* Tabs */}
        <Paper>
          <Tabs value={tabValue} onChange={(_, v) => setTabValue(v)}>
            <Tab label="Active Alerts" />
            <Tab label="Alert History" />
          </Tabs>

          {/* Active Alerts Tab */}
          <TabPanel value={tabValue} index={0}>
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
                  {filteredAlerts.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={9} align="center" sx={{ py: 6 }}>
                        <Typography color="text.secondary">
                          {searchQuery || filterType !== 'all' ? 'No alerts match your filters' : 'No active alerts. Create one to get started.'}
                        </Typography>
                        {!searchQuery && filterType === 'all' && (
                          <Button variant="contained" sx={{ mt: 2 }} startIcon={<Add />}
                            onClick={() => { setForm(EMPTY_FORM); setCreateOpen(true); }}>
                            Create Your First Alert
                          </Button>
                        )}
                      </TableCell>
                    </TableRow>
                  ) : filteredAlerts.map((a) => (
                    <TableRow key={a.id} hover>
                      <TableCell><Typography fontWeight="bold">{a.ticker}</Typography></TableCell>
                      <TableCell><TypeCell type={a.type} /></TableCell>
                      <TableCell>
                        <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
                          {a.type === 'news' ? '-' : a.condition}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2">{a.type === 'news' ? '-' : formatAlertValue(a.type, a.value)}</Typography>
                      </TableCell>
                      <TableCell align="center">
                        <Chip label={a.status.charAt(0).toUpperCase() + a.status.slice(1)} color={STATUS_COLORS[a.status]} size="small" />
                      </TableCell>
                      <TableCell align="center">
                        <NotifChips inApp={a.notifyInApp} email={a.notifyEmail} />
                      </TableCell>
                      <TableCell><Typography variant="body2">{formatDate(a.createdAt)}</Typography></TableCell>
                      <TableCell>
                        <Typography variant="body2" color={isExpiringSoon(a.expiresAt) ? 'warning.main' : 'text.secondary'}>
                          {a.expiresAt ? formatDate(a.expiresAt) : 'Never'}
                        </Typography>
                      </TableCell>
                      <TableCell align="center">
                        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                          <Tooltip title={a.status === 'active' ? 'Pause alert' : a.status === 'paused' ? 'Resume alert' : 'Cannot toggle'}>
                            <span>
                              <IconButton size="small" color={a.status === 'active' ? 'primary' : 'default'}
                                onClick={() => handleToggle(a.id)} disabled={a.status === 'expired' || a.status === 'triggered'}>
                                {a.status === 'active' ? <NotificationsActive fontSize="small" /> : <NotificationsOff fontSize="small" />}
                              </IconButton>
                            </span>
                          </Tooltip>
                          <IconButton size="small" onClick={(e) => handleMenuOpen(e, a.id)}>
                            <MoreVert fontSize="small" />
                          </IconButton>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </TabPanel>

          {/* Alert History Tab */}
          <TabPanel value={tabValue} index={1}>
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
                  {filteredAlerts.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={5} align="center" sx={{ py: 6 }}>
                        <Typography color="text.secondary">No triggered alerts to display</Typography>
                      </TableCell>
                    </TableRow>
                  ) : filteredAlerts.map((a) => (
                    <TableRow key={a.id} hover>
                      <TableCell><Typography fontWeight="bold">{a.ticker}</Typography></TableCell>
                      <TableCell><TypeCell type={a.type} /></TableCell>
                      <TableCell><Typography variant="body2">{a.message}</Typography></TableCell>
                      <TableCell><Typography variant="body2">{formatDate(a.triggeredAt)}</Typography></TableCell>
                      <TableCell align="center"><NotifChips inApp={a.notifyInApp} email={a.notifyEmail} /></TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </TabPanel>
        </Paper>
      </Box>

      {/* Context Menu */}
      <Menu anchorEl={anchorEl} open={Boolean(anchorEl)} onClose={handleMenuClose}>
        <MenuItem onClick={() => { if (menuAlert) openEdit(menuAlert); }}>
          <Edit sx={{ mr: 1 }} fontSize="small" /> Edit
        </MenuItem>
        <MenuItem onClick={() => { if (menuId) { handleToggle(menuId); handleMenuClose(); } }}>
          <NotificationsOff sx={{ mr: 1 }} fontSize="small" />
          {menuAlert?.status === 'active' ? 'Pause' : 'Resume'}
        </MenuItem>
        <MenuItem onClick={() => { if (menuId) { handleDelete(menuId); handleMenuClose(); } }}>
          <Delete sx={{ mr: 1 }} fontSize="small" color="error" /> Delete
        </MenuItem>
      </Menu>

      {/* Create Alert Dialog */}
      <Dialog open={createOpen} onClose={() => setCreateOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create Alert</DialogTitle>
        <DialogContent>{renderForm()}</DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateOpen(false)}>Cancel</Button>
          <Button onClick={handleCreate} variant="contained" disabled={formDisabled}>Create</Button>
        </DialogActions>
      </Dialog>

      {/* Edit Alert Dialog */}
      <Dialog open={editOpen} onClose={() => setEditOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Edit Alert - {editingAlert?.ticker}</DialogTitle>
        <DialogContent>{renderForm()}</DialogContent>
        <DialogActions>
          <Button onClick={() => setEditOpen(false)}>Cancel</Button>
          <Button onClick={handleEdit} variant="contained" disabled={formDisabled}>Save Changes</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default Alerts;
