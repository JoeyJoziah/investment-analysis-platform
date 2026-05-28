import React, { useEffect, useRef, useState } from 'react';
import { Snackbar, Alert, type AlertColor } from '@mui/material';
import { useAppSelector } from '../../hooks/redux';

interface ActiveToast {
  id: string;
  type: AlertColor;
  message: string;
}

const AUTO_HIDE_MS = 5000;

// Surfaces each new app notification as a transient, auto-dismissing toast.
// NotificationPanel (the bell) is a persistent log; this component pops every
// new notification immediately so actions like "Save" give visible feedback.
// Notifications are left in state (the bell log keeps them); we only track
// which ids we've already toasted so each is shown exactly once, oldest first.
const NotificationToaster: React.FC = () => {
  const { notifications } = useAppSelector((state) => state.app);
  const shownIds = useRef<Set<string>>(new Set());
  const seeded = useRef(false);
  const [active, setActive] = useState<ActiveToast | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    // On first run, treat any pre-existing notifications as already seen so a
    // backlog (e.g. from a page refresh) isn't replayed as a burst of toasts.
    if (!seeded.current) {
      notifications.forEach((n) => shownIds.current.add(n.id));
      seeded.current = true;
      return;
    }
    if (active) {
      return;
    }
    const next = notifications.find((n) => !shownIds.current.has(n.id));
    if (next) {
      shownIds.current.add(next.id);
      setActive({ id: next.id, type: next.type, message: next.message });
      setOpen(true);
    }
  }, [notifications, active]);

  const handleClose = (_event?: React.SyntheticEvent | Event, reason?: string) => {
    if (reason === 'clickaway') {
      return;
    }
    setOpen(false);
  };

  // Clear the active slot only after the exit transition so the next pending
  // notification surfaces (the effect re-runs when `active` becomes null).
  const handleExited = () => {
    setActive(null);
  };

  return (
    <Snackbar
      open={open}
      autoHideDuration={AUTO_HIDE_MS}
      onClose={handleClose}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      TransitionProps={{ onExited: handleExited }}
    >
      {active ? (
        <Alert
          onClose={handleClose}
          severity={active.type}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {active.message}
        </Alert>
      ) : undefined}
    </Snackbar>
  );
};

export default NotificationToaster;
