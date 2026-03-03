import React, { useState, useMemo, useCallback } from 'react';
import {
  Container, Grid, Paper, Typography, Box, Card, CardContent,
  Button, Tabs, Tab, TextField, MenuItem, FormControl, InputLabel,
  Select, InputAdornment,
} from '@mui/material';
import {
  Add, NotificationsActive, TrendingUp, Search, FilterList, AccessTime,
} from '@mui/icons-material';
import { useAppDispatch } from '../hooks/redux';
import { addNotification } from '../store/slices/appSlice';
import AlertsList, {
  type AlertItem,
  type AlertStatus,
  buildMessage,
  isExpiringSoon,
} from '../components/alerts/AlertsList';
import AlertForm, {
  type AlertType,
  type AlertCondition,
  type AlertFormData,
  EMPTY_FORM,
} from '../components/alerts/AlertForm';

// --- Tab panel helper ---

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel({ children, value, index, ...other }: TabPanelProps) {
  return (
    <div hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

// --- Summary card helper ---

const SummaryCard: React.FC<{
  label: string;
  count: number;
  color: string;
  icon: React.ReactNode;
}> = ({ label, count, color, icon }) => (
  <Grid item xs={12} sm={4}>
    <Card>
      <CardContent>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <Box>
            <Typography color="text.secondary" gutterBottom variant="caption">
              {label}
            </Typography>
            <Typography variant="h4" fontWeight="bold" color={`${color}.main`}>
              {count}
            </Typography>
          </Box>
          {icon}
        </Box>
      </CardContent>
    </Card>
  </Grid>
);

// --- Seed data ---

const mkAlert = (
  id: string,
  ticker: string,
  type: AlertType,
  condition: AlertCondition,
  value: number,
  status: AlertStatus,
  notifyEmail: boolean,
  createdAt: string,
  triggeredAt: string | null,
  expiresAt: string | null,
  message: string
): AlertItem => ({
  id,
  ticker,
  type,
  condition,
  value,
  status,
  notifyInApp: true,
  notifyEmail,
  createdAt,
  triggeredAt,
  expiresAt,
  message,
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

// --- Component ---

const Alerts: React.FC = () => {
  const dispatch = useAppDispatch();
  const [alerts, setAlerts] = useState<AlertItem[]>(INITIAL_ALERTS);
  const [tabValue, setTabValue] = useState(0);
  const [createOpen, setCreateOpen] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [form, setForm] = useState<AlertFormData>(EMPTY_FORM);
  const [editingAlert, setEditingAlert] = useState<AlertItem | null>(null);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [menuId, setMenuId] = useState<string | null>(null);
  const [filterType, setFilterType] = useState<AlertType | 'all'>('all');
  const [searchQuery, setSearchQuery] = useState('');

  // Derived data
  const activeAlerts = useMemo(
    () => alerts.filter((a) => a.status === 'active'),
    [alerts]
  );
  const triggeredToday = useMemo(() => {
    const start = new Date();
    start.setHours(0, 0, 0, 0);
    return alerts.filter(
      (a) =>
        a.status === 'triggered' &&
        a.triggeredAt &&
        new Date(a.triggeredAt) >= start
    );
  }, [alerts]);
  const expiringSoon = useMemo(
    () => alerts.filter((a) => a.status === 'active' && isExpiringSoon(a.expiresAt)),
    [alerts]
  );

  const filteredAlerts = useMemo(() => {
    const base =
      tabValue === 0
        ? alerts.filter((a) => a.status !== 'triggered')
        : alerts.filter((a) => a.status === 'triggered');
    return base
      .filter((a) => filterType === 'all' || a.type === filterType)
      .filter(
        (a) =>
          a.ticker.toLowerCase().includes(searchQuery.toLowerCase()) ||
          a.message.toLowerCase().includes(searchQuery.toLowerCase())
      );
  }, [alerts, tabValue, filterType, searchQuery]);

  // Handlers
  const notify = useCallback(
    (type: 'success' | 'info' | 'error', message: string) => {
      dispatch(addNotification({ type, message }));
    },
    [dispatch]
  );

  const handleCreate = useCallback(() => {
    if (!form.ticker.trim() || (form.type !== 'news' && !form.value)) return;
    const val = form.type === 'news' ? 0 : Number(form.value);
    const ticker = form.ticker.toUpperCase();
    const newAlert: AlertItem = {
      id: `alert-${Date.now()}`,
      ticker,
      type: form.type,
      condition: form.condition,
      value: val,
      status: 'active',
      notifyInApp: form.notifyInApp,
      notifyEmail: form.notifyEmail,
      createdAt: new Date().toISOString(),
      triggeredAt: null,
      expiresAt: null,
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
    setAlerts((prev) =>
      prev.map((a) =>
        a.id === editingAlert.id
          ? {
              ...a,
              ticker,
              type: form.type,
              condition: form.condition,
              value: val,
              notifyInApp: form.notifyInApp,
              notifyEmail: form.notifyEmail,
              message: buildMessage(form.type, ticker, form.condition, val),
            }
          : a
      )
    );
    setEditOpen(false);
    setEditingAlert(null);
    setForm(EMPTY_FORM);
    notify('success', `Alert updated for ${ticker}`);
  }, [editingAlert, form, notify]);

  const handleDelete = useCallback(
    (id: string) => {
      const target = alerts.find((a) => a.id === id);
      setAlerts((prev) => prev.filter((a) => a.id !== id));
      notify('info', `Alert removed${target ? ` for ${target.ticker}` : ''}`);
    },
    [alerts, notify]
  );

  const handleToggle = useCallback(
    (id: string) => {
      const target = alerts.find((a) => a.id === id);
      setAlerts((prev) =>
        prev.map((a) => {
          if (a.id !== id) return a;
          return {
            ...a,
            status: (a.status === 'active' ? 'paused' : 'active') as AlertStatus,
          };
        })
      );
      if (target) {
        notify(
          'info',
          `Alert ${target.status === 'active' ? 'paused' : 'resumed'} for ${target.ticker}`
        );
      }
    },
    [alerts, notify]
  );

  const openEditDialog = useCallback((alert: AlertItem) => {
    setEditingAlert(alert);
    setForm({
      ticker: alert.ticker,
      type: alert.type,
      condition: alert.condition,
      value: alert.type === 'news' ? '' : String(alert.value),
      notifyInApp: alert.notifyInApp,
      notifyEmail: alert.notifyEmail,
    });
    setEditOpen(true);
    setAnchorEl(null);
    setMenuId(null);
  }, []);

  const handleMenuOpen = useCallback(
    (e: React.MouseEvent<HTMLElement>, id: string) => {
      setAnchorEl(e.currentTarget);
      setMenuId(id);
    },
    []
  );

  const handleMenuClose = useCallback(() => {
    setAnchorEl(null);
    setMenuId(null);
  }, []);

  const handleOpenCreate = useCallback(() => {
    setForm(EMPTY_FORM);
    setCreateOpen(true);
  }, []);

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 3 }}>
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            mb: 2,
          }}
        >
          <Box>
            <Typography variant="h4" fontWeight="bold">
              Alerts
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Monitor price movements, volume spikes, and portfolio drift
            </Typography>
          </Box>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={handleOpenCreate}
          >
            Create Alert
          </Button>
        </Box>

        {/* Summary Cards */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <SummaryCard
            label="Total Active"
            count={activeAlerts.length}
            color="success"
            icon={
              <NotificationsActive
                sx={{ fontSize: 40, color: 'success.main', opacity: 0.3 }}
              />
            }
          />
          <SummaryCard
            label="Triggered Today"
            count={triggeredToday.length}
            color="warning"
            icon={
              <TrendingUp
                sx={{ fontSize: 40, color: 'warning.main', opacity: 0.3 }}
              />
            }
          />
          <SummaryCard
            label="Expiring Soon"
            count={expiringSoon.length}
            color="error"
            icon={
              <AccessTime
                sx={{ fontSize: 40, color: 'error.main', opacity: 0.3 }}
              />
            }
          />
        </Grid>

        {/* Search and Filter */}
        <Paper sx={{ p: 2, mb: 3 }}>
          <Box
            sx={{
              display: 'flex',
              gap: 2,
              alignItems: 'center',
              flexWrap: 'wrap',
            }}
          >
            <TextField
              placeholder="Search alerts..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              sx={{ minWidth: 250, flexGrow: 1 }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search />
                  </InputAdornment>
                ),
              }}
            />
            <FormControl sx={{ minWidth: 160 }}>
              <InputLabel>Alert Type</InputLabel>
              <Select
                value={filterType}
                label="Alert Type"
                onChange={(e) =>
                  setFilterType(e.target.value as AlertType | 'all')
                }
                startAdornment={
                  <InputAdornment position="start">
                    <FilterList />
                  </InputAdornment>
                }
              >
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

        {/* Tabs + Lists */}
        <Paper>
          <Tabs value={tabValue} onChange={(_, v) => setTabValue(v)}>
            <Tab label="Active Alerts" />
            <Tab label="Alert History" />
          </Tabs>

          <TabPanel value={tabValue} index={0}>
            <AlertsList
              alerts={filteredAlerts}
              tabValue={0}
              searchQuery={searchQuery}
              filterType={filterType}
              anchorEl={anchorEl}
              menuId={menuId}
              onToggle={handleToggle}
              onDelete={handleDelete}
              onEdit={openEditDialog}
              onMenuOpen={handleMenuOpen}
              onMenuClose={handleMenuClose}
              onCreateOpen={handleOpenCreate}
            />
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            <AlertsList
              alerts={filteredAlerts}
              tabValue={1}
              searchQuery={searchQuery}
              filterType={filterType}
              anchorEl={null}
              menuId={null}
              onToggle={handleToggle}
              onDelete={handleDelete}
              onEdit={openEditDialog}
              onMenuOpen={handleMenuOpen}
              onMenuClose={handleMenuClose}
              onCreateOpen={handleOpenCreate}
            />
          </TabPanel>
        </Paper>
      </Box>

      {/* Create Alert Dialog */}
      <AlertForm
        open={createOpen}
        form={form}
        mode="create"
        onFormChange={setForm}
        onSubmit={handleCreate}
        onClose={() => setCreateOpen(false)}
      />

      {/* Edit Alert Dialog */}
      <AlertForm
        open={editOpen}
        form={form}
        mode="edit"
        editTicker={editingAlert?.ticker}
        onFormChange={setForm}
        onSubmit={handleEdit}
        onClose={() => setEditOpen(false)}
      />
    </Container>
  );
};

export default Alerts;
