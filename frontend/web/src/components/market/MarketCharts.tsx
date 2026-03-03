/**
 * Market Charts - Sector performance chart, heatmap, and economic calendar
 * Renders the Sectors, Heat Map, and Economic Calendar tab panels
 */

import React from 'react';
import {
  Grid,
  Typography,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
} from '@mui/material';
import {
  BarChartOutlined,
  GridViewOutlined,
  EventNoteOutlined,
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { format } from 'date-fns';
import MarketHeatmap from '../charts/MarketHeatmap';
import { EmptyStateBox } from './MarketSummary';
import type { SectorPerformance } from '../../store/slices/marketSlice';

interface HeatmapItem {
  ticker: string;
  name: string;
  sector: string;
  changePercent: number;
  marketCap: number;
  volume: number;
}

interface EconomicEvent {
  date: string;
  time: string;
  event: string;
  importance: 'high' | 'medium' | 'low';
  actual?: number;
  forecast?: number;
  previous?: number;
}

const SectorPanel: React.FC<{
  sectorPerformance: SectorPerformance[];
  formatPercent: (value: number) => string;
}> = ({ sectorPerformance, formatPercent }) => {
  if (sectorPerformance.length === 0) {
    return (
      <EmptyStateBox
        icon={<BarChartOutlined sx={{ fontSize: 48 }} />}
        message="No sector performance data available"
        submessage="Sector data will appear here once connected to a data provider."
        minHeight={300}
      />
    );
  }

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} md={8}>
        <Typography variant="h6" gutterBottom>
          Sector Performance
        </Typography>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={sectorPerformance}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="sector" angle={-45} textAnchor="end" height={80} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="changePercent" fill="#8884d8">
              {sectorPerformance.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.changePercent >= 0 ? '#00C49F' : '#FF8042'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Grid>
      <Grid item xs={12} md={4}>
        <Typography variant="h6" gutterBottom>
          Sector Details
        </Typography>
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Sector</TableCell>
                <TableCell align="right">Change</TableCell>
                <TableCell align="center">Leaders</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {sectorPerformance.map((sector) => (
                <TableRow key={sector.sector}>
                  <TableCell>{sector.sector}</TableCell>
                  <TableCell align="right">
                    <Typography
                      variant="body2"
                      color={sector.changePercent >= 0 ? 'success.main' : 'error.main'}
                    >
                      {formatPercent(sector.changePercent)}
                    </Typography>
                  </TableCell>
                  <TableCell align="center">
                    <Box>
                      <Typography variant="caption">
                        {sector.topStock.ticker}
                      </Typography>
                      <Typography
                        variant="caption"
                        display="block"
                        color={sector.topStock.changePercent >= 0 ? 'success.main' : 'error.main'}
                      >
                        {formatPercent(sector.topStock.changePercent)}
                      </Typography>
                    </Box>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Grid>
    </Grid>
  );
};

const HeatmapPanel: React.FC<{ heatmapData: HeatmapItem[] }> = ({ heatmapData }) => {
  return (
    <>
      <Typography variant="h6" gutterBottom>
        Market Heat Map
      </Typography>
      {heatmapData.length > 0 ? (
        <Box sx={{ height: 600 }}>
          <MarketHeatmap
            data={heatmapData.map((item) => ({
              name: item.name,
              ticker: item.ticker,
              value: item.marketCap,
              change: item.changePercent,
              volume: item.volume,
              sector: item.sector,
            }))}
          />
        </Box>
      ) : (
        <EmptyStateBox
          icon={<GridViewOutlined sx={{ fontSize: 48 }} />}
          message="No heatmap data available"
          submessage="The market heatmap will appear here once connected to a data provider."
          minHeight={300}
        />
      )}
    </>
  );
};

const EconomicCalendarPanel: React.FC<{ economicCalendar: EconomicEvent[] }> = ({ economicCalendar }) => {
  return (
    <>
      <Typography variant="h6" gutterBottom>
        Economic Calendar
      </Typography>
      {economicCalendar.length > 0 ? (
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Date/Time</TableCell>
                <TableCell>Event</TableCell>
                <TableCell align="center">Importance</TableCell>
                <TableCell align="right">Actual</TableCell>
                <TableCell align="right">Forecast</TableCell>
                <TableCell align="right">Previous</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {economicCalendar.map((event, index) => (
                <TableRow key={index}>
                  <TableCell>
                    <Typography variant="body2">
                      {format(new Date(event.date), 'MMM dd')}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {event.time}
                    </Typography>
                  </TableCell>
                  <TableCell>{event.event}</TableCell>
                  <TableCell align="center">
                    <Chip
                      label={event.importance}
                      size="small"
                      color={
                        event.importance === 'high'
                          ? 'error'
                          : event.importance === 'medium'
                          ? 'warning'
                          : 'default'
                      }
                    />
                  </TableCell>
                  <TableCell align="right">
                    <Typography
                      variant="body2"
                      fontWeight={event.actual ? 'bold' : 'normal'}
                      color={
                        event.actual && event.forecast
                          ? event.actual > event.forecast
                            ? 'success.main'
                            : event.actual < event.forecast
                            ? 'error.main'
                            : 'text.primary'
                          : 'text.primary'
                      }
                    >
                      {event.actual || '-'}
                    </Typography>
                  </TableCell>
                  <TableCell align="right">{event.forecast || '-'}</TableCell>
                  <TableCell align="right">{event.previous || '-'}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      ) : (
        <EmptyStateBox
          icon={<EventNoteOutlined sx={{ fontSize: 48 }} />}
          message="No economic calendar events available"
          submessage="Upcoming economic events will appear here once connected to a data provider."
          minHeight={300}
        />
      )}
    </>
  );
};

export { SectorPanel, HeatmapPanel, EconomicCalendarPanel };
