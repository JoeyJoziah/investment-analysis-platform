import React, { useState, useMemo, useCallback } from 'react';
import {
  Container, Grid, Paper, Typography, Box, Card, CardContent, CardActionArea,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Button, IconButton, Chip, Tabs, Tab, TextField, MenuItem,
  FormControl, InputLabel, Select, Switch, FormControlLabel,
  LinearProgress, Tooltip, Alert, alpha, useTheme,
} from '@mui/material';
import {
  Assessment, Description, Download, Delete, Visibility, Schedule, Storage,
  TrendingUp, AccountBalance, Receipt, LocalOffer, AttachMoney, Security,
  PlayArrow, PictureAsPdf, TableChart, GridOn,
} from '@mui/icons-material';
import { useAppDispatch } from '../hooks/redux';
import { addNotification } from '../store/slices/appSlice';

// --- Types ---

interface TabPanelProps {
  readonly children?: React.ReactNode;
  readonly index: number;
  readonly value: number;
}

type ReportStatus = 'ready' | 'processing' | 'failed';
type ReportFormat = 'PDF' | 'CSV' | 'Excel';

interface ReportType {
  readonly id: string;
  readonly title: string;
  readonly description: string;
  readonly icon: React.ReactNode;
  readonly color: string;
}

interface RecentReport {
  readonly id: string;
  readonly name: string;
  readonly type: string;
  readonly format: ReportFormat;
  readonly status: ReportStatus;
  readonly generatedAt: string;
  readonly fileSize: string;
  readonly dateRange: string;
}

interface ScheduledReport {
  readonly id: string;
  readonly name: string;
  readonly type: string;
  readonly frequency: string;
  readonly delivery: string;
  readonly enabled: boolean;
  readonly nextRun: string;
  readonly lastRun: string;
}

interface GeneratorForm {
  readonly reportType: string;
  readonly startDate: string;
  readonly endDate: string;
  readonly format: ReportFormat;
}

// --- Constants ---

const REPORT_TYPES: readonly ReportType[] = [
  { id: 'portfolio-performance', title: 'Portfolio Performance', description: 'Returns, benchmarks, and attribution analysis', icon: <TrendingUp />, color: '#0088FE' },
  { id: 'holdings-summary', title: 'Holdings Summary', description: 'Positions, market values, cost basis, unrealized gains', icon: <AccountBalance />, color: '#00C49F' },
  { id: 'transaction-history', title: 'Transaction History', description: 'All buys, sells, dividends, and corporate actions', icon: <Receipt />, color: '#FFBB28' },
  { id: 'tax-loss-harvesting', title: 'Tax Loss Harvesting', description: 'Offset capital gains with realized losses', icon: <LocalOffer />, color: '#FF8042' },
  { id: 'dividend-income', title: 'Dividend Income', description: 'Payments received, yield analysis, projected income', icon: <AttachMoney />, color: '#8884d8' },
  { id: 'risk-analysis', title: 'Risk Analysis', description: 'VaR, drawdown, correlation, and stress tests', icon: <Security />, color: '#82ca9d' },
];

const INITIAL_RECENT: readonly RecentReport[] = [
  { id: 'rpt-001', name: 'Portfolio Performance - Q4 2025', type: 'portfolio-performance', format: 'PDF', status: 'ready', generatedAt: '2026-02-28T14:30:00Z', fileSize: '2.4 MB', dateRange: 'Oct 1 - Dec 31, 2025' },
  { id: 'rpt-002', name: 'Holdings Summary - Feb 2026', type: 'holdings-summary', format: 'Excel', status: 'ready', generatedAt: '2026-02-27T09:15:00Z', fileSize: '1.1 MB', dateRange: 'Feb 1 - 28, 2026' },
  { id: 'rpt-003', name: 'Transaction History - YTD', type: 'transaction-history', format: 'CSV', status: 'processing', generatedAt: '2026-03-01T08:00:00Z', fileSize: '--', dateRange: 'Jan 1 - Mar 1, 2026' },
  { id: 'rpt-004', name: 'Tax Loss Harvesting - 2025', type: 'tax-loss-harvesting', format: 'PDF', status: 'ready', generatedAt: '2026-01-15T11:45:00Z', fileSize: '3.8 MB', dateRange: 'Jan 1 - Dec 31, 2025' },
  { id: 'rpt-005', name: 'Risk Analysis - Feb 2026', type: 'risk-analysis', format: 'PDF', status: 'failed', generatedAt: '2026-02-25T16:20:00Z', fileSize: '--', dateRange: 'Feb 1 - 25, 2026' },
  { id: 'rpt-006', name: 'Dividend Income - 2025', type: 'dividend-income', format: 'Excel', status: 'ready', generatedAt: '2026-01-10T10:00:00Z', fileSize: '0.9 MB', dateRange: 'Jan 1 - Dec 31, 2025' },
];

const INITIAL_SCHEDULED: readonly ScheduledReport[] = [
  { id: 'sch-001', name: 'Monthly Portfolio Performance', type: 'portfolio-performance', frequency: 'monthly', delivery: 'email', enabled: true, nextRun: '2026-04-01T08:00:00Z', lastRun: '2026-03-01T08:00:00Z' },
  { id: 'sch-002', name: 'Weekly Holdings Summary', type: 'holdings-summary', frequency: 'weekly', delivery: 'download', enabled: true, nextRun: '2026-03-07T08:00:00Z', lastRun: '2026-02-28T08:00:00Z' },
  { id: 'sch-003', name: 'Quarterly Tax Report', type: 'tax-loss-harvesting', frequency: 'quarterly', delivery: 'email', enabled: false, nextRun: '2026-07-01T08:00:00Z', lastRun: '2026-01-01T08:00:00Z' },
];

const STATUS_CONFIG: Record<ReportStatus, { label: string; color: 'success' | 'warning' | 'error' }> = {
  ready: { label: 'Ready', color: 'success' },
  processing: { label: 'Processing', color: 'warning' },
  failed: { label: 'Failed', color: 'error' },
};

const FORMAT_ICONS: Record<ReportFormat, React.ReactNode> = {
  PDF: <PictureAsPdf fontSize="small" />,
  CSV: <TableChart fontSize="small" />,
  Excel: <GridOn fontSize="small" />,
};

// --- Helpers ---

function TabPanel({ children, value, index, ...other }: TabPanelProps) {
  return (
    <div hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const fmtDate = (iso: string) =>
  new Date(iso).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });

const fmtShort = (iso: string) =>
  new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });

const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

// --- Stat Card ---

function StatCard({ label, value, sub, icon, color }: {
  label: string; value: string | number; sub: string; icon: React.ReactNode; color: string;
}) {
  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box>
            <Typography color="text.secondary" gutterBottom variant="caption">{label}</Typography>
            <Typography variant="h5" fontWeight="bold">{value}</Typography>
            <Typography variant="body2" color="text.secondary">{sub}</Typography>
          </Box>
          <Box sx={{ backgroundColor: color, borderRadius: 2, p: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
}

// --- Main Component ---

const Reports: React.FC = () => {
  const theme = useTheme();
  const dispatch = useAppDispatch();
  const [tabValue, setTabValue] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [recentReports, setRecentReports] = useState<readonly RecentReport[]>(INITIAL_RECENT);
  const [scheduledReports, setScheduledReports] = useState<readonly ScheduledReport[]>(INITIAL_SCHEDULED);
  const [form, setForm] = useState<GeneratorForm>({ reportType: '', startDate: '2026-02-01', endDate: '2026-03-01', format: 'PDF' });

  // Computed
  const reportsThisMonth = useMemo(() => {
    const now = new Date();
    return recentReports.filter((r) => {
      const d = new Date(r.generatedAt);
      return d.getMonth() === now.getMonth() && d.getFullYear() === now.getFullYear();
    }).length;
  }, [recentReports]);

  const nextScheduled = useMemo(() => {
    const enabled = scheduledReports.filter((s) => s.enabled);
    if (enabled.length === 0) return 'None';
    const sorted = [...enabled].sort((a, b) => new Date(a.nextRun).getTime() - new Date(b.nextRun).getTime());
    return fmtShort(sorted[0].nextRun);
  }, [scheduledReports]);

  const storageUsed = useMemo(() => {
    return recentReports
      .filter((r) => r.status === 'ready')
      .reduce((sum, r) => { const m = r.fileSize.match(/^([\d.]+)\s*MB$/); return m ? sum + parseFloat(m[1]) : sum; }, 0)
      .toFixed(1);
  }, [recentReports]);

  // Handlers
  const updateField = useCallback(<K extends keyof GeneratorForm>(field: K, value: GeneratorForm[K]) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  }, []);

  const handleGenerate = useCallback(() => {
    if (!form.reportType) {
      dispatch(addNotification({ type: 'warning', message: 'Please select a report type' }));
      return;
    }
    setIsGenerating(true);
    const label = REPORT_TYPES.find((rt) => rt.id === form.reportType)?.title ?? form.reportType;

    setTimeout(() => {
      const newReport: RecentReport = {
        id: `rpt-${Date.now()}`, name: `${label} - Custom`, type: form.reportType,
        format: form.format, status: 'processing', generatedAt: new Date().toISOString(),
        fileSize: '--', dateRange: `${fmtShort(form.startDate)} - ${fmtShort(form.endDate)}`,
      };
      setRecentReports((prev) => [newReport, ...prev]);
      setIsGenerating(false);
      setTabValue(1);
      dispatch(addNotification({ type: 'success', message: `${label} report is being generated` }));

      setTimeout(() => {
        setRecentReports((prev) => prev.map((r) =>
          r.id === newReport.id ? { ...r, status: 'ready' as const, fileSize: '1.7 MB' } : r
        ));
      }, 3000);
    }, 1500);
  }, [form, dispatch]);

  const notify = useCallback((type: 'success' | 'info' | 'warning' | 'error', message: string) => {
    dispatch(addNotification({ type, message }));
  }, [dispatch]);

  const handleDeleteReport = useCallback((id: string) => {
    setRecentReports((prev) => prev.filter((r) => r.id !== id));
    notify('info', 'Report deleted');
  }, [notify]);

  const handleDownload = useCallback((r: RecentReport) => {
    notify('success', `Downloading ${r.name}.${r.format.toLowerCase()}`);
  }, [notify]);

  const handleView = useCallback((r: RecentReport) => {
    notify('info', `Opening ${r.name} preview`);
  }, [notify]);

  const handleToggleSchedule = useCallback((id: string) => {
    setScheduledReports((prev) => prev.map((s) => (s.id === id ? { ...s, enabled: !s.enabled } : s)));
  }, []);

  const handleDeleteSchedule = useCallback((id: string) => {
    setScheduledReports((prev) => prev.filter((s) => s.id !== id));
    notify('info', 'Scheduled report removed');
  }, [notify]);

  return (
    <Container maxWidth="xl">
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight="bold">Reports</Typography>
        <Typography variant="body2" color="text.secondary">Generate, schedule, and manage your investment reports</Typography>
      </Box>

      {/* Summary Stats */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={4}>
          <StatCard label="Reports Generated" value={reportsThisMonth} sub="This month"
            icon={<Assessment sx={{ fontSize: 32, color: 'primary.main' }} />}
            color={alpha(theme.palette.primary.main, 0.1)} />
        </Grid>
        <Grid item xs={12} sm={4}>
          <StatCard label="Next Scheduled Report" value={nextScheduled}
            sub={`${scheduledReports.filter((s) => s.enabled).length} active schedules`}
            icon={<Schedule sx={{ fontSize: 32, color: 'info.main' }} />}
            color={alpha(theme.palette.info.main, 0.1)} />
        </Grid>
        <Grid item xs={12} sm={4}>
          <StatCard label="Storage Used" value={`${storageUsed} MB`}
            sub={`${recentReports.filter((r) => r.status === 'ready').length} files stored`}
            icon={<Storage sx={{ fontSize: 32, color: 'warning.main' }} />}
            color={alpha(theme.palette.warning.main, 0.1)} />
        </Grid>
      </Grid>

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={tabValue} onChange={(_, v) => setTabValue(v)}>
          <Tab icon={<Description />} label="Generate" iconPosition="start" />
          <Tab icon={<Assessment />} label="Recent Reports" iconPosition="start" />
          <Tab icon={<Schedule />} label="Scheduled Reports" iconPosition="start" />
        </Tabs>

        {/* Generate Tab */}
        <TabPanel value={tabValue} index={0}>
          <Typography variant="h6" gutterBottom>Select Report Type</Typography>
          <Grid container spacing={2} sx={{ mb: 4 }}>
            {REPORT_TYPES.map((rt) => {
              const selected = form.reportType === rt.id;
              return (
                <Grid item xs={12} sm={6} md={4} key={rt.id}>
                  <Card variant={selected ? 'outlined' : 'elevation'}
                    sx={{ border: selected ? 2 : undefined, borderColor: selected ? 'primary.main' : undefined, transition: 'border 0.2s' }}>
                    <CardActionArea onClick={() => updateField('reportType', rt.id)} sx={{ p: 2 }}>
                      <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
                        <Box sx={{ backgroundColor: alpha(rt.color, 0.15), borderRadius: 2, p: 1, display: 'flex', color: rt.color }}>
                          {rt.icon}
                        </Box>
                        <Box sx={{ flex: 1, minWidth: 0 }}>
                          <Typography variant="subtitle2" fontWeight="bold">{rt.title}</Typography>
                          <Typography variant="caption" color="text.secondary">{rt.description}</Typography>
                        </Box>
                      </Box>
                    </CardActionArea>
                  </Card>
                </Grid>
              );
            })}
          </Grid>

          <Typography variant="h6" gutterBottom>Configure Report</Typography>
          <Paper variant="outlined" sx={{ p: 3 }}>
            <Grid container spacing={3} alignItems="center">
              <Grid item xs={12} sm={6} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Report Type</InputLabel>
                  <Select value={form.reportType} label="Report Type" onChange={(e) => updateField('reportType', e.target.value)}>
                    {REPORT_TYPES.map((rt) => <MenuItem key={rt.id} value={rt.id}>{rt.title}</MenuItem>)}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <TextField fullWidth label="Start Date" type="date" value={form.startDate}
                  onChange={(e) => updateField('startDate', e.target.value)} InputLabelProps={{ shrink: true }} />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <TextField fullWidth label="End Date" type="date" value={form.endDate}
                  onChange={(e) => updateField('endDate', e.target.value)} InputLabelProps={{ shrink: true }} />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Format</InputLabel>
                  <Select value={form.format} label="Format" onChange={(e) => updateField('format', e.target.value as ReportFormat)}>
                    <MenuItem value="PDF">PDF</MenuItem>
                    <MenuItem value="CSV">CSV</MenuItem>
                    <MenuItem value="Excel">Excel</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Button variant="contained" startIcon={isGenerating ? undefined : <PlayArrow />}
                    onClick={handleGenerate} disabled={isGenerating || !form.reportType} size="large">
                    {isGenerating ? 'Generating...' : 'Generate Report'}
                  </Button>
                  {isGenerating && <LinearProgress sx={{ flex: 1, maxWidth: 200 }} />}
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </TabPanel>

        {/* Recent Reports Tab */}
        <TabPanel value={tabValue} index={1}>
          {recentReports.length === 0 ? (
            <Alert severity="info">No reports generated yet. Use the Generate tab to create one.</Alert>
          ) : (
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Report Name</TableCell>
                    <TableCell>Date Range</TableCell>
                    <TableCell>Format</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell align="right">Size</TableCell>
                    <TableCell>Generated</TableCell>
                    <TableCell align="center">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {recentReports.map((r) => {
                    const sc = STATUS_CONFIG[r.status];
                    return (
                      <TableRow key={r.id}>
                        <TableCell><Typography variant="subtitle2" fontWeight="bold">{r.name}</Typography></TableCell>
                        <TableCell><Typography variant="body2" color="text.secondary">{r.dateRange}</Typography></TableCell>
                        <TableCell>
                          <Chip icon={FORMAT_ICONS[r.format] as React.ReactElement} label={r.format} size="small" variant="outlined" />
                        </TableCell>
                        <TableCell><Chip label={sc.label} color={sc.color} size="small" /></TableCell>
                        <TableCell align="right"><Typography variant="body2">{r.fileSize}</Typography></TableCell>
                        <TableCell><Typography variant="body2" color="text.secondary">{fmtDate(r.generatedAt)}</Typography></TableCell>
                        <TableCell align="center">
                          <Tooltip title="View"><span>
                            <IconButton size="small" onClick={() => handleView(r)} disabled={r.status !== 'ready'}><Visibility fontSize="small" /></IconButton>
                          </span></Tooltip>
                          <Tooltip title="Download"><span>
                            <IconButton size="small" onClick={() => handleDownload(r)} disabled={r.status !== 'ready'}><Download fontSize="small" /></IconButton>
                          </span></Tooltip>
                          <Tooltip title="Delete">
                            <IconButton size="small" color="error" onClick={() => handleDeleteReport(r.id)}><Delete fontSize="small" /></IconButton>
                          </Tooltip>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </TabPanel>

        {/* Scheduled Reports Tab */}
        <TabPanel value={tabValue} index={2}>
          {scheduledReports.length === 0 ? (
            <Alert severity="info">No scheduled reports configured.</Alert>
          ) : (
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Report Name</TableCell>
                    <TableCell>Frequency</TableCell>
                    <TableCell>Delivery</TableCell>
                    <TableCell>Next Run</TableCell>
                    <TableCell>Last Run</TableCell>
                    <TableCell align="center">Enabled</TableCell>
                    <TableCell align="center">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {scheduledReports.map((s) => (
                    <TableRow key={s.id} sx={{ opacity: s.enabled ? 1 : 0.6 }}>
                      <TableCell><Typography variant="subtitle2" fontWeight="bold">{s.name}</Typography></TableCell>
                      <TableCell><Chip label={capitalize(s.frequency)} size="small" variant="outlined" color="primary" /></TableCell>
                      <TableCell><Chip label={capitalize(s.delivery)} size="small" variant="outlined" /></TableCell>
                      <TableCell><Typography variant="body2" color="text.secondary">{fmtShort(s.nextRun)}</Typography></TableCell>
                      <TableCell><Typography variant="body2" color="text.secondary">{fmtShort(s.lastRun)}</Typography></TableCell>
                      <TableCell align="center">
                        <FormControlLabel label=""
                          control={<Switch checked={s.enabled} onChange={() => handleToggleSchedule(s.id)} size="small" />} />
                      </TableCell>
                      <TableCell align="center">
                        <Tooltip title="Delete schedule">
                          <IconButton size="small" color="error" onClick={() => handleDeleteSchedule(s.id)}><Delete fontSize="small" /></IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default Reports;
