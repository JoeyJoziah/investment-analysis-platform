import { useNavigate, Link as RouterLink } from 'react-router-dom';
import { Container, Paper, Box, Typography, Button, Link } from '@mui/material';
import { SentimentDissatisfied, Home } from '@mui/icons-material';

/**
 * 404 "Page Not Found" page.
 *
 * Rendered by the authenticated catch-all route when a user navigates to an
 * unknown URL, instead of silently redirecting to the dashboard.
 */
function NotFound() {
  const navigate = useNavigate();

  return (
    <Container component="main" maxWidth="sm">
      <Box
        sx={{
          marginTop: 8,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        <Paper
          elevation={3}
          sx={{
            padding: 4,
            width: '100%',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            textAlign: 'center',
          }}
        >
          <SentimentDissatisfied
            sx={{ fontSize: 64, color: 'primary.main', mb: 2 }}
            aria-hidden="true"
          />
          <Typography component="h1" variant="h3" fontWeight="bold" gutterBottom>
            404
          </Typography>
          <Typography variant="h5" fontWeight="bold" gutterBottom>
            Page Not Found
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            The page you are looking for does not exist or may have been moved.
            Check the URL or head back to your dashboard.
          </Typography>

          <Button
            variant="contained"
            startIcon={<Home />}
            onClick={() => navigate('/dashboard')}
          >
            Back to Dashboard
          </Button>

          <Box sx={{ mt: 2 }}>
            <Link component={RouterLink} to="/help" variant="body2" underline="hover">
              Visit the Help Center
            </Link>
          </Box>
        </Paper>
      </Box>
    </Container>
  );
}

export default NotFound;
