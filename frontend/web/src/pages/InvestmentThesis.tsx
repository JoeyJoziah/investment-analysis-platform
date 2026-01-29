import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Button,
  TextField,
  MenuItem,
  Alert,
  CircularProgress,
  IconButton,
  Divider,
  Stack,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Save,
  Download,
  Upload,
  History,
  Description,
  ArrowBack,
  Info,
} from '@mui/icons-material';
import axios from 'axios';

// NOTE: Monaco Editor needs to be installed
// Run: npm install @monaco-editor/react
// Import will be: import Editor from '@monaco-editor/react';

interface InvestmentThesis {
  id?: number;
  stock_id: number;
  investment_objective: string;
  time_horizon: string;
  target_price: number | null;
  business_model: string | null;
  competitive_advantages: string | null;
  financial_health: string | null;
  growth_drivers: string | null;
  risks: string | null;
  valuation_rationale: string | null;
  exit_strategy: string | null;
  content: string | null;
  version?: number;
  created_at?: string;
  updated_at?: string;
  stock_symbol?: string;
  stock_name?: string;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

const InvestmentThesisPage: React.FC = () => {
  const { stockId } = useParams<{ stockId: string }>();
  const navigate = useNavigate();

  // State
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [thesis, setThesis] = useState<InvestmentThesis | null>(null);
  const [templateDialogOpen, setTemplateDialogOpen] = useState(false);
  const [markdownContent, setMarkdownContent] = useState('');

  // Form state for structured fields
  const [objective, setObjective] = useState('');
  const [timeHorizon, setTimeHorizon] = useState<'short-term' | 'medium-term' | 'long-term'>('medium-term');
  const [targetPrice, setTargetPrice] = useState<string>('');

  // Load existing thesis
  useEffect(() => {
    if (stockId) {
      loadThesis();
    }
  }, [stockId]);

  const loadThesis = async () => {
    if (!stockId) return;

    setLoading(true);
    setError(null);

    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(
        `${API_BASE_URL}/thesis/stock/${stockId}`,
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );

      const data = response.data;
      setThesis(data);
      setObjective(data.investment_objective || '');
      setTimeHorizon(data.time_horizon || 'medium-term');
      setTargetPrice(data.target_price?.toString() || '');
      setMarkdownContent(data.content || '');
    } catch (err: any) {
      if (err.response?.status === 404) {
        // No thesis exists yet - that's okay
        setError(null);
      } else {
        setError('Failed to load investment thesis');
        console.error('Error loading thesis:', err);
      }
    } finally {
      setLoading(false);
    }
  };

  const loadTemplate = async () => {
    try {
      // Load template from the markdown file
      const response = await fetch('/docs/templates/investment_thesis_template.md');
      const template = await response.text();
      setMarkdownContent(template);
      setTemplateDialogOpen(false);
      setSuccess('Template loaded successfully');
    } catch (err) {
      setError('Failed to load template');
      console.error('Error loading template:', err);
    }
  };

  const saveThesis = async () => {
    if (!stockId || !objective || !timeHorizon) {
      setError('Please fill in required fields: Investment Objective and Time Horizon');
      return;
    }

    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      const token = localStorage.getItem('token');
      const thesisData: Partial<InvestmentThesis> = {
        stock_id: parseInt(stockId),
        investment_objective: objective,
        time_horizon: timeHorizon,
        target_price: targetPrice ? parseFloat(targetPrice) : null,
        content: markdownContent,
      };

      let response;
      if (thesis?.id) {
        // Update existing thesis
        response = await axios.put(
          `${API_BASE_URL}/thesis/${thesis.id}`,
          thesisData,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );
      } else {
        // Create new thesis
        response = await axios.post(
          `${API_BASE_URL}/thesis/`,
          thesisData,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );
      }

      setThesis(response.data);
      setSuccess('Investment thesis saved successfully');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save investment thesis');
      console.error('Error saving thesis:', err);
    } finally {
      setSaving(false);
    }
  };

  const exportAsMarkdown = () => {
    if (!markdownContent) {
      setError('No content to export');
      return;
    }

    const blob = new Blob([markdownContent], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `investment_thesis_${thesis?.stock_symbol || 'stock'}_${new Date().toISOString().split('T')[0]}.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    setSuccess('Thesis exported as Markdown');
  };

  const exportAsPDF = () => {
    // TODO: Implement PDF export
    // This would require a library like jsPDF or html2pdf
    setError('PDF export not yet implemented');
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Box mb={3}>
        <Stack direction="row" alignItems="center" spacing={2} mb={2}>
          <IconButton onClick={() => navigate(-1)}>
            <ArrowBack />
          </IconButton>
          <Typography variant="h4" component="h1">
            Investment Thesis
            {thesis?.stock_symbol && ` - ${thesis.stock_symbol}`}
            {thesis?.stock_name && ` (${thesis.stock_name})`}
          </Typography>
        </Stack>

        {thesis && (
          <Stack direction="row" spacing={1}>
            <Chip label={`Version ${thesis.version}`} size="small" />
            <Chip
              label={`Updated ${new Date(thesis.updated_at!).toLocaleDateString()}`}
              size="small"
              variant="outlined"
            />
          </Stack>
        )}
      </Box>

      {/* Alerts */}
      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      {success && (
        <Alert severity="success" onClose={() => setSuccess(null)} sx={{ mb: 2 }}>
          {success}
        </Alert>
      )}

      {/* Main Content */}
      <Grid container spacing={3}>
        {/* Sidebar - Structured Fields */}
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Core Details
            </Typography>

            <TextField
              fullWidth
              label="Investment Objective *"
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              margin="normal"
              multiline
              rows={3}
              helperText="Primary investment goal"
            />

            <TextField
              fullWidth
              select
              label="Time Horizon *"
              value={timeHorizon}
              onChange={(e) => setTimeHorizon(e.target.value as any)}
              margin="normal"
            >
              <MenuItem value="short-term">Short-term (0-1 year)</MenuItem>
              <MenuItem value="medium-term">Medium-term (1-5 years)</MenuItem>
              <MenuItem value="long-term">Long-term (5+ years)</MenuItem>
            </TextField>

            <TextField
              fullWidth
              label="Target Price"
              value={targetPrice}
              onChange={(e) => setTargetPrice(e.target.value)}
              margin="normal"
              type="number"
              inputProps={{ step: '0.01', min: '0' }}
              helperText="Price target based on valuation"
            />

            <Divider sx={{ my: 2 }} />

            <Stack spacing={1}>
              <Button
                fullWidth
                variant="contained"
                startIcon={<Save />}
                onClick={saveThesis}
                disabled={saving}
              >
                {saving ? 'Saving...' : thesis?.id ? 'Update Thesis' : 'Create Thesis'}
              </Button>

              <Button
                fullWidth
                variant="outlined"
                startIcon={<Description />}
                onClick={() => setTemplateDialogOpen(true)}
              >
                Load Template
              </Button>

              <Button
                fullWidth
                variant="outlined"
                startIcon={<Download />}
                onClick={exportAsMarkdown}
                disabled={!markdownContent}
              >
                Export as Markdown
              </Button>

              <Button
                fullWidth
                variant="outlined"
                startIcon={<Download />}
                onClick={exportAsPDF}
                disabled={!markdownContent}
              >
                Export as PDF
              </Button>
            </Stack>
          </Paper>

          {/* Info Panel */}
          <Paper sx={{ p: 2, mt: 2 }}>
            <Stack direction="row" spacing={1} alignItems="flex-start">
              <Info color="primary" fontSize="small" />
              <Typography variant="caption" color="text.secondary">
                The markdown editor below supports full markdown syntax. Use the template
                to get started with a comprehensive structure.
              </Typography>
            </Stack>
          </Paper>
        </Grid>

        {/* Main Editor */}
        <Grid item xs={12} md={9}>
          <Paper sx={{ p: 3, minHeight: '70vh' }}>
            <Typography variant="h6" gutterBottom>
              Thesis Document (Markdown)
            </Typography>

            {/* Monaco Editor Placeholder */}
            <Box
              sx={{
                border: '1px solid',
                borderColor: 'divider',
                borderRadius: 1,
                minHeight: '60vh',
                bgcolor: 'background.default',
                p: 2,
              }}
            >
              <TextField
                fullWidth
                multiline
                rows={30}
                value={markdownContent}
                onChange={(e) => setMarkdownContent(e.target.value)}
                placeholder="Write your investment thesis in markdown format..."
                variant="outlined"
                sx={{
                  '& .MuiInputBase-root': {
                    fontFamily: 'monospace',
                    fontSize: '0.9rem',
                  },
                }}
              />

              <Alert severity="info" sx={{ mt: 2 }}>
                <Typography variant="body2">
                  <strong>To enable rich markdown editing:</strong>
                  <br />
                  1. Install Monaco Editor: <code>npm install @monaco-editor/react</code>
                  <br />
                  2. Replace the TextField above with the Monaco Editor component
                  <br />
                  3. This will provide syntax highlighting, autocomplete, and a better
                  editing experience
                </Typography>
              </Alert>
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Load Template Dialog */}
      <Dialog open={templateDialogOpen} onClose={() => setTemplateDialogOpen(false)}>
        <DialogTitle>Load Investment Thesis Template</DialogTitle>
        <DialogContent>
          <Typography>
            This will load a comprehensive investment thesis template with sections for:
          </Typography>
          <ul>
            <li>Executive Summary</li>
            <li>Business Model Analysis</li>
            <li>Competitive Advantages (Moats)</li>
            <li>Financial Health Assessment</li>
            <li>Growth Drivers</li>
            <li>Risk Assessment (Bear/Base/Bull Cases)</li>
            <li>Valuation Analysis</li>
            <li>Investment & Exit Strategy</li>
            <li>Decision Log</li>
          </ul>
          <Alert severity="warning" sx={{ mt: 2 }}>
            This will replace your current content. Make sure to save any existing work first.
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTemplateDialogOpen(false)}>Cancel</Button>
          <Button onClick={loadTemplate} variant="contained">
            Load Template
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default InvestmentThesisPage;
