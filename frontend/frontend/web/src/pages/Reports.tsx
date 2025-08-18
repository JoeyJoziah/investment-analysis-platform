import React from 'react';
import { Container, Typography, Box, Paper } from '@mui/material';

const Reports: React.FC = () => {
  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight="bold">
          Reports
        </Typography>
      </Box>
      <Paper sx={{ p: 3 }}>
        <Typography>Reports page - Coming soon</Typography>
      </Paper>
    </Container>
  );
};

export default Reports;