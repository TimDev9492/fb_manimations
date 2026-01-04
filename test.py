from manimlib import *
import numpy as np

from manim_slides import Slide

class MyScene(Scene):
    def construct(self):
        c = Circle()
        c.set_fill(BLUE, opacity=0.5)
        c.set_stroke(YELLOW_A, width=4)

        self.play(FadeIn(c))

class CoordinateSystemExample(Scene):
    def construct(self):
        axes = Axes(
            # x-axis ranges from -1 to 10, with a default step size of 1
            x_range=(-1, 10),
            # y-axis ranges from -2 to 2 with a step size of 0.5
            y_range=(-2, 2, 0.5),
            # The axes will be stretched so as to match the specified
            # height and width
            height=6,
            width=10,
            # Axes is made of two NumberLine mobjects.  You can specify
            # their configuration with axis_config
            axis_config={
                "stroke_color": GREY_A,
                "stroke_width": 2,
            },
            # Alternatively, you can specify configuration for just one
            # of them, like this.
            y_axis_config={
                "include_tip": False,
            }
        )
        # Keyword arguments of add_coordinate_labels can be used to
        # configure the DecimalNumber mobjects which it creates and
        # adds to the axes
        axes.add_coordinate_labels(
            font_size=20,
            num_decimal_places=1,
        )
        self.add(axes)

        # Axes descends from the CoordinateSystem class, meaning
        # you can call call axes.coords_to_point, abbreviated to
        # axes.c2p, to associate a set of coordinates with a point,
        # like so:
        dot = Dot(color=RED)
        dot.move_to(axes.c2p(0, 0))
        self.play(FadeIn(dot, scale=0.5))
        self.play(dot.animate.move_to(axes.c2p(3, 2)))
        self.wait()
        self.play(dot.animate.move_to(axes.c2p(5, 0.5)))
        self.wait()

        # Similarly, you can call axes.point_to_coords, or axes.p2c
        # print(axes.p2c(dot.get_center()))

        # We can draw lines from the axes to better mark the coordinates
        # of a given point.
        # Here, the always_redraw command means that on each new frame
        # the lines will be redrawn
        h_line = always_redraw(lambda: axes.get_h_line(dot.get_left()))
        v_line = always_redraw(lambda: axes.get_v_line(dot.get_bottom()))

        self.play(
            ShowCreation(h_line),
            ShowCreation(v_line),
        )
        self.play(dot.animate.move_to(axes.c2p(3, -2)))
        self.wait()
        self.play(dot.animate.move_to(axes.c2p(1, 1)))
        self.wait()

        # If we tie the dot to a particular set of coordinates, notice
        # that as we move the axes around it respects the coordinate
        # system defined by them.
        f_always(dot.move_to, lambda: axes.c2p(1, 1))
        self.play(
            axes.animate.scale(0.75).to_corner(UL),
            run_time=2,
        )
        self.wait()
        self.play(FadeOut(VGroup(axes, dot, h_line, v_line)))

        # Other coordinate systems you can play around with include
        # ThreeDAxes, NumberPlane, and ComplexPlane.

class BinomialTree(Slide):
    def setup(self):
        self.T = 0.5
        self.n = 3
        self.delta_t = self.T / self.n
        self.riskfree_rate = 0.06
        self.asset_volatility = 0.2
        self.asset_price = 100
        self.strike_price = 95
        self.tilt = 0
        self.discrete_rate_factor = np.exp(self.riskfree_rate * self.delta_t)
        
        self.u = np.exp(self.asset_volatility * np.sqrt(self.delta_t) + self.tilt * self.asset_volatility**2 * self.delta_t)
        self.d = np.exp(-self.asset_volatility * np.sqrt(self.delta_t) + self.tilt * self.asset_volatility**2 * self.delta_t)

        self.y_min = self.asset_price * self.d**self.n
        self.y_max = self.asset_price * self.u**self.n

    def S(self, i, j):
        return self.asset_price * self.u**j * self.d**(i-j)

    def construct(self):
        self.setup()

        # axes = Axes(
        #     x_range=(0, self.T, self.delta_t),
        #     y_range=(self.y_min, self.y_max, (self.y_max - self.y_min) / self.n),
        #     # y_range=(self.y_min, self.y_max, 1),
        #     height=6,
        #     width=10
        # )
        # axes.add_coordinate_labels(
        #     font_size=20,
        #     num_decimal_places=2
        # )

        price_range = (self.y_min, self.y_max, (self.y_max - self.y_min) / self.n)
        t_range = (0, self.T, self.delta_t)

        price_axis = NumberLine(
            x_range=price_range,
            width=6,
            include_ticks=False,
            decimal_number_config={
                'font_size': 20,
                'num_decimal_places': 2,
            },
            line_to_number_direction=LEFT
        )
        price_axis.rotate(90 * DEG, about_point=ORIGIN)

        prices = [
            self.S(i, j) for i in (self.n - 1, self.n) for j in range(self.n+1)
        ]

        for price in prices:
            ticks = VGroup()
            ticks.add(price_axis.get_tick(price, price_axis.tick_size))
            price_axis.add(ticks)

        price_axis.add_numbers(prices)
        price_axis.to_edge(LEFT)
        self.play(ShowCreation(price_axis))

        self.next_slide()

        t_axis = NumberLine(
            x_range=t_range,
            width=10,
            decimal_number_config={
                'font_size': 20,
                'num_decimal_places': 2,
            },
            # unit_size=self.delta_t
        )
        t_axis.add_numbers()
        t_axis.to_edge(DOWN)
        self.play(ShowCreation(t_axis))

        def c2p(t, S):
            t_aligned = t_axis.n2p(t)
            t_aligned[1] = price_axis.n2p(S)[1]
            return t_aligned
            
        self.next_slide()

        S_0 = Dot(c2p(0, self.asset_price), fill_color=YELLOW_B)
        label = Tex('S_0 = 100', font_size=20)
        label.next_to(S_0, DOWN)

        self.play(GrowFromCenter(S_0), Write(label))

        self.next_slide()

        self.play(FadeOut(label))

        

