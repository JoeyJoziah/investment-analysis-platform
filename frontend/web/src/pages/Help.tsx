import React from 'react';
import { Container, Typography, Box, Paper } from '@mui/material';

const Help: React.FC = () => {
  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight="bold">
          Help & Support
        </Typography>
      </Box>
      <Paper sx={{ p: 3 }}>
        <Typography>Help page - Coming soon</Typography>
      </Paper>
    </Container>
  );
};

export default Help;