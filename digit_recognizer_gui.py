from gi.repository import Gtk, Gdk, GdkPixbuf


def recognizer():
    class RecognizerWindow(Gtk.Window):
        def __init__(self):
            Gtk.Window.__init__(self, title="Digit Recognizer")
            self.recognize_button = Gtk.Button("Recognize")
            self.clear_button = Gtk.Button("Clear")
            self.exit_button = Gtk.Button("Exit")

            self.recognize_button.connect('clicked', self.on_recognize_clicked)
            self.clear_button.connect('clicked', self.on_clear_clicked)
            self.exit_button.connect('clicked', Gtk.main_quit)
            self.connect('delete-event', Gtk.main_quit)

            #
            # Drawing area
            #
            self.drawing_area = Gtk.DrawingArea()
            self.drawing_area.set_size_request(28*4, 28*4)
            self.drawing_area.add_events(Gdk.EventMask.BUTTON_PRESS_MASK
                                         | Gdk.EventMask.POINTER_MOTION_MASK
                                         | Gdk.EventMask.POINTER_MOTION_HINT_MASK
                                         | Gdk.EventMask.LEAVE_NOTIFY_MASK)
            self.drawing_area.connect('draw', self.on_drawing_area_draw)
            self.drawing_area.connect('button-press-event', self.on_drawing_area_button_pressed)
            self.drawing_area.connect('motion-notify-event', self.on_drawing_area_motion_notify)
            self.drawing_area.connect('configure-event', self.on_drawing_area_configure)
            self.drawing_area_clicks = []
            self.drawing_position_reset = True
            self.drawing_area_image = None

            #
            # Stats labels
            #
            self.result_label = Gtk.Label("None")
            self.stats_labels = [Gtk.Label("{} : ".format(i)) for i in range(10)]
            self.stats_results = [Gtk.Label("0.000000") for i in range(10)]

            #
            # Objects placement
            #

            # Main buttons
            self.button_box = Gtk.VBox()
            self.button_box.pack_start(self.recognize_button, False, False, 3)
            self.button_box.pack_start(self.clear_button, False, False, 3)
            self.button_box.pack_start(self.exit_button, False, False, 3)

            # Stats labels
            self.stats_table = Gtk.Grid()
            self.stats_table.add(self.result_label)
            for i in range(10):
                self.stats_table.attach_next_to(self.stats_labels[i], self.result_label if i == 0 else self.stats_labels[i-1], Gtk.PositionType.BOTTOM, 1, 1)
                self.stats_table.attach_next_to(self.stats_results[i], self.stats_labels[i], Gtk.PositionType.RIGHT, 1, 1)

            # Final placement
            self.main_box = Gtk.Box()
            self.add(self.main_box)

            self.main_box.pack_start(self.drawing_area, False, False, 3)
            self.main_box.pack_start(self.stats_table, False, False, 3)
            self.main_box.pack_start(self.button_box, False, False, 3)

            #
            # Final operations
            #
            self.show_all()

        def on_drawing_area_configure(self, widget, event):
            if not self.drawing_area_image:
                self.drawing_area_image = Gdk.PixBuf(widget.get_window(), 28*4, 28*4)
                # Gtk.draw_rectangle(self.drawing_area_pixmap, widget.get_style().white_gc, True, 0, 0, 28*4, 28*4)
            return True

        def on_drawing_area_draw(self, widget, cairo_context):
            print("draw")
            for x, y in self.drawing_area_clicks:
                if x < 0:
                    x = -x
                    cairo_context.move_to(x-1, y)
                cairo_context.line_to(x, y)
            cairo_context.set_source_rgb(0, 0, 0)
            cairo_context.stroke()
            return False

        def on_drawing_area_motion_notify(self, widget, event):
            print("motion")
            if event.is_hint:
                state = 0
                print(event.window.get_pointer())
                _, x, y, state = event.window.get_pointer()
                # x, y = event.window.get_pointer
                # state = event.window.pointer_state
            else:
                x, y = event.x, event.y
                state = event.state

            if state & state.BUTTON1_MASK:
                if self.drawing_position_reset:
                    x = -x
                    self.drawing_position_reset = False
                self.drawing_area_clicks.append((x, y))
                self.drawing_area.queue_draw()
            else:
                self.drawing_position_reset = True
            return True

        def on_drawing_area_button_pressed(self, widget, event):
            print("click")
            self.drawing_area_clicks.append((-event.x, event.y))
            self.drawing_position_reset = False
            self.drawing_area.queue_draw()
            return True

        def on_clear_clicked(self, widget):
            print("clear")

        def on_recognize_clicked(self, widget):
            print("recognize")

    window = RecognizerWindow()
    Gtk.main()
