import React from 'react';
import { Container, Typography, Box, Paper } from '@mui/material';

const Alerts: React.FC = () => {
  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight="bold">
          Alerts
        </Typography>
      </Box>
      <Paper sx={{ p: 3 }}>
        <Typography>Alerts page - Coming soon</Typography>
      </Paper>
    </Container>
  );
};

export default Alerts;